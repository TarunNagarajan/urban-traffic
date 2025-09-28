import React, { useState, useEffect, useRef, useMemo } from 'react';
import { SumoMap } from './SumoMap';

// --- Types to match the new CSV structure ---
type VehicleData = {
  simulation_time_s: number;
  vehicle_id: string;
  x_coord_m: number;
  y_coord_m: number;
  current_intersection_id: string | null;
  intersection_queue_length_veh: number | null;
  waiting_time_s: number;
};

type Mode = 'LIVE' | 'PLAYBACK';

// --- Constants ---
const API_BASE_URL = 'http://localhost:8000';
const WS_URL = 'ws://localhost:8000/ws/simulation-stream';

// --- Helper Components ---
type ControlButtonProps = {
  onClick: React.MouseEventHandler<HTMLButtonElement>;
  disabled?: boolean;
  children: React.ReactNode;
};

const ControlButton: React.FC<ControlButtonProps> = ({ onClick, disabled, children }) => (
  <button onClick={onClick} disabled={disabled} style={{ margin: '0 5px' }}>
    {children}
  </button>
);

const TimeSlider = ({ time, maxTime, onTimeChange }) => (
  <input
    type="range"
    min="0"
    max={maxTime}
    value={time}
    onChange={(e) => onTimeChange(Number(e.target.value))}
    style={{ width: '100%' }}
  />
);

const AnalysisPanel = ({ vehicles }: { vehicles: VehicleData[] }) => {
    const vehiclesAtIntersection = vehicles.filter(v => v.current_intersection_id && v.current_intersection_id !== '');
    const intersectionData = vehiclesAtIntersection[0];

    return (
        <div>
            <h3>Analysis</h3>
            {intersectionData ? (
                <>
                    <p><strong>Intersection ID:</strong> {intersectionData.current_intersection_id}</p>
                    <p><strong>Queue Length:</strong> {intersectionData.intersection_queue_length_veh} vehicles</p>
                    <p><strong>Max Waiting Time:</strong> {Math.max(...vehicles.map(v => v.waiting_time_s)).toFixed(2)}s</p>
                </>
            ) : (
                <p>No vehicles at an intersection currently.</p>
            )}
            <p><strong>Vehicles at Intersection:</strong> {vehiclesAtIntersection.length}</p>
            <p><strong>Total Visible Vehicles:</strong> {vehicles.length}</p>
        </div>
    );
};

// --- Main Dashboard Component ---
export function Dashboard(): JSX.Element {
  const [mode, setMode] = useState<Mode>('PLAYBACK');
  const [historicalData, setHistoricalData] = useState<VehicleData[]>([]);
  const [liveData, setLiveData] = useState<VehicleData | null>(null);
  const [currentTimeStep, setCurrentTimeStep] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [playbackSpeed, setPlaybackSpeed] = useState(1);
  const ws = useRef<WebSocket | null>(null);

  const timeSteps = useMemo(() => {
      const uniqueTimes = [...new Set(historicalData.map(d => d.simulation_time_s))];
      return uniqueTimes.sort((a, b) => a - b);
  }, [historicalData]);

  // Fetch historical data
  useEffect(() => {
    fetch(`${API_BASE_URL}/api/historical-data`)
      .then((res) => res.json())
      .then((data) => setHistoricalData(data))
      .catch(console.error);
  }, []);

  // WebSocket for live mode
  useEffect(() => {
    if (mode === 'LIVE') {
      ws.current = new WebSocket(WS_URL);
      ws.current.onmessage = (event) => {
        const newData = JSON.parse(event.data);
        setLiveData(newData);
        const timeIndex = timeSteps.indexOf(newData.simulation_time_s);
        if (timeIndex !== -1) setCurrentTimeStep(timeIndex);
      };
      return () => ws.current?.close();
    }
  }, [mode, timeSteps]);

  // Playback timer
  useEffect(() => {
    if (mode === 'PLAYBACK' && isPlaying && timeSteps.length > 0) {
      const interval = setInterval(() => {
        setCurrentTimeStep((prev) => (prev + 1) % timeSteps.length);
      }, 1000 / playbackSpeed);
      return () => clearInterval(interval);
    }
  }, [isPlaying, playbackSpeed, mode, timeSteps]);

  const vehiclesAtCurrentTime = useMemo(() => {
    const currentSimTime = timeSteps[currentTimeStep];
    if (mode === 'LIVE' && liveData) {
        // In a true live scenario, you might want to accumulate data
        return [liveData];
    }
    return historicalData.filter(d => d.simulation_time_s === currentSimTime);
  }, [mode, liveData, currentTimeStep, historicalData, timeSteps]);

  const intersectionData = useMemo(() => {
    const intersectionsMap = new Map<string, { id: string; phase: number; totalQueue: number }>();
    vehiclesAtCurrentTime.forEach(vehicle => {
      if (vehicle.current_intersection_id && vehicle.current_intersection_id !== '') {
        if (!intersectionsMap.has(vehicle.current_intersection_id)) {
          intersectionsMap.set(vehicle.current_intersection_id, {
            id: vehicle.current_intersection_id,
            phase: 0, // Mock phase
            totalQueue: vehicle.intersection_queue_length_veh || 0,
          });
        }
      }
    });
    return Array.from(intersectionsMap.values());
  }, [vehiclesAtCurrentTime]);

  return (
    <div style={{ padding: '20px' }}>
      <h1>SUMO Traffic Simulation Dashboard</h1>
      <div style={{ display: 'grid', gridTemplateColumns: '3fr 1fr', gap: '20px' }}>
        <div>
          <h2>Simulation View</h2>
          <SumoMap vehicles={vehiclesAtCurrentTime} width={800} height={600} />
        </div>
        <div>
          <h2>Controls & Analysis</h2>
          <div style={{ border: '1px solid #ddd', padding: '15px', borderRadius: '8px' }}>
            <button onClick={() => setMode(m => m === 'LIVE' ? 'PLAYBACK' : 'LIVE')}>
              Switch to {mode === 'LIVE' ? 'Playback' : 'Live'} Mode
            </button>
            <hr style={{ margin: '15px 0' }} />
            <h3>{mode} Mode</h3>
            {mode === 'PLAYBACK' && (
              <>
                <TimeSlider 
                  time={currentTimeStep} 
                  maxTime={timeSteps.length - 1} 
                  onTimeChange={setCurrentTimeStep} 
                />
                <p>Time: {timeSteps[currentTimeStep]}s</p>
                <div>
                  <ControlButton onClick={() => setIsPlaying(!isPlaying)} disabled={timeSteps.length === 0}>
                    {isPlaying ? 'Pause' : 'Play'}
                  </ControlButton>
                  <label>
                    Speed:
                    <select value={playbackSpeed} onChange={(e) => setPlaybackSpeed(Number(e.target.value))}>
                      <option value={0.5}>0.5x</option>
                      <option value={1}>1x</option>
                      <option value={2}>2x</option>
                    </select>
                  </label>
                </div>
              </>
            )}
            {mode === 'LIVE' && <p>Watching live data stream...</p>}
          </div>
          <div style={{ marginTop: '20px', border: '1px solid #ddd', padding: '15px', borderRadius: '8px' }}>
            <AnalysisPanel vehicles={vehiclesAtCurrentTime} />
          </div>
        </div>
      </div>
    </div>
  );
}
