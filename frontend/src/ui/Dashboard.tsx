import React, { useEffect, useMemo, useRef, useState } from 'react'
import { IntersectionVisualization } from './IntersectionVisualization'

type Intersection = {
  id: string
  phase: number
  totalQueue: number
}

type StreamPayload = {
  done: boolean
  intersections: Intersection[]
}

const WS_URL = (location.protocol === 'https:' ? 'wss://' : 'ws://') + (location.hostname + ':8000') + '/ws'

export function Dashboard(): JSX.Element {
  const [connected, setConnected] = useState(false)
  const [running, setRunning] = useState(false)
  const [demoMode, setDemoMode] = useState(false)
  const [csvMode, setCsvMode] = useState(false)
  const [data, setData] = useState<StreamPayload | null>(null)
  const wsRef = useRef<WebSocket | null>(null)
  const demoTimerRef = useRef<number | null>(null)
  const csvTimerRef = useRef<number | null>(null)
  const csvFramesRef = useRef<StreamPayload[]>([])

  const startSim = async () => {
    await fetch('http://localhost:8000/start', { method: 'POST' })
    setRunning(true)
  }

  const stopSim = async () => {
    await fetch('http://localhost:8000/stop', { method: 'POST' })
    setRunning(false)
  }

  // Live backend stream
  useEffect(() => {
    if (demoMode || csvMode) return
    const ws = new WebSocket(WS_URL)
    wsRef.current = ws
    ws.onopen = () => setConnected(true)
    ws.onclose = () => setConnected(false)
    ws.onerror = () => setConnected(false)
    ws.onmessage = (event) => {
      const payload = JSON.parse(event.data) as StreamPayload
      setData(payload)
      if (payload.done) setRunning(false)
    }
    return () => ws.close()
  }, [demoMode, csvMode])

  // Demo mode generator
  useEffect(() => {
    if (!demoMode) {
      if (demoTimerRef.current) window.clearInterval(demoTimerRef.current)
      return
    }
    setConnected(true)
    setRunning(true)
    let step = 0
    let intersections: Intersection[] = [
      { id: 'gneJ0', phase: 0, totalQueue: 3 },
      { id: 'gneJ1', phase: 1, totalQueue: 8 },
      { id: 'gneJ2', phase: 0, totalQueue: 4 },
      { id: 'gneJ3', phase: 1, totalQueue: 6 },
    ]
    const tick = () => {
      step += 1
      intersections = intersections.map((i) => {
        const delta = Math.round((Math.random() - 0.4) * 3)
        const newQueue = Math.max(0, i.totalQueue + delta)
        const switchPhase = Math.random() < 0.25
        return {
          ...i,
          totalQueue: newQueue,
          phase: switchPhase ? (i.phase === 0 ? 1 : 0) : i.phase,
        }
      })
      setData({ done: false, intersections })
    }
    tick()
    demoTimerRef.current = window.setInterval(tick, 1000)
    return () => {
      if (demoTimerRef.current) window.clearInterval(demoTimerRef.current)
    }
  }, [demoMode])

  // CSV playback mode
  useEffect(() => {
    if (!csvMode) {
      if (csvTimerRef.current) window.clearInterval(csvTimerRef.current)
      return
    }
    setConnected(true)
    setRunning(!!csvFramesRef.current.length)
    let idx = 0
    const tick = () => {
      if (!csvFramesRef.current.length) return
      setData(csvFramesRef.current[idx])
      idx = (idx + 1) % csvFramesRef.current.length
    }
    tick()
    csvTimerRef.current = window.setInterval(tick, 1000)
    return () => {
      if (csvTimerRef.current) window.clearInterval(csvTimerRef.current)
    }
  }, [csvMode])

  const onPickCsv = async (file: File) => {
    const text = await file.text()
    // Basic CSV parsing (comma separated). Details CSV format:
    // step,reward, <ts_id>_phase, <ts_id>_total_queue, ...
    const lines = text.trim().split(/\r?\n/)
    if (lines.length < 2) return
    const header = lines[0].split(',').map((h) => h.trim())
    const tsIds: string[] = []
    for (let i = 0; i < header.length; i++) {
      const h = header[i]
      if (h.endsWith('_total_queue')) {
        tsIds.push(h.replace('_total_queue', ''))
      }
    }
    const frames: StreamPayload[] = []
    for (let li = 1; li < lines.length; li++) {
      const cols = lines[li].split(',')
      const intersections: Intersection[] = tsIds.map((id) => {
        const phaseIdx = header.indexOf(id + '_phase')
        const queueIdx = header.indexOf(id + '_total_queue')
        const phase = phaseIdx >= 0 ? Number(cols[phaseIdx] ?? 0) : 0
        const totalQueue = queueIdx >= 0 ? Number(cols[queueIdx] ?? 0) : 0
        return { id, phase, totalQueue }
      })
      frames.push({ done: false, intersections })
    }
    csvFramesRef.current = frames
    setCsvMode(true)
    setDemoMode(false)
    setRunning(!!frames.length)
  }

  const totalQueue = useMemo(() => {
    if (!data) return 0
    return data.intersections.reduce((acc, i) => acc + i.totalQueue, 0)
  }, [data])

  return (
    <div style={{ fontFamily: 'Inter, system-ui, Arial', padding: 16, lineHeight: 1.4 }}>
      <h2 style={{ margin: 0 }}>Urban Traffic Simulator</h2>
      <div style={{ marginTop: 8, display: 'flex', gap: 12, alignItems: 'center' }}>
        <span>Server: {connected ? 'Connected' : 'Disconnected'}</span>
        <button onClick={startSim} disabled={demoMode || csvMode || !connected || running}>Start</button>
        <button onClick={stopSim} disabled={demoMode || csvMode || !connected || !running}>Stop</button>
        <span style={{ marginLeft: 8, color: '#666' }}>|</span>
        <button
          onClick={() => setDemoMode((v) => !v)}
          title="Toggle demo data (no backend required)"
        >
          {demoMode ? 'Exit Demo' : 'Demo Mode'}
        </button>
        <label style={{ display: 'inline-flex', gap: 8, alignItems: 'center' }}>
          <span style={{ color: '#666' }}>| CSV Playback:</span>
          <input
            type="file"
            accept=".csv"
            onChange={(e) => {
              const f = e.target.files?.[0]
              if (f) onPickCsv(f)
            }}
          />
          {csvMode && (
            <button onClick={() => { setCsvMode(false); setRunning(false) }}>Stop CSV</button>
          )}
        </label>
      </div>

      <div style={{ marginTop: 16, display: 'grid', gridTemplateColumns: '1fr 2fr', gap: 16 }}>
        <div style={{ border: '1px solid #ddd', padding: 12, borderRadius: 8 }}>
          <h3 style={{ marginTop: 0 }}>System Metrics</h3>
          <div>Intersections: {data?.intersections.length ?? 0}</div>
          <div>Total Queue: {totalQueue.toFixed(0)}</div>
        </div>

        <div style={{ border: '1px solid #ddd', padding: 12, borderRadius: 8 }}>
          <h3 style={{ marginTop: 0 }}>Intersections</h3>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(180px, 1fr))', gap: 12 }}>
            {data?.intersections.map((i) => (
              <div key={i.id} style={{ border: '1px solid #eee', borderRadius: 8, padding: 10 }}>
                <div style={{ fontWeight: 600, marginBottom: 6 }}>{i.id}</div>
                <div>Phase: {i.phase}</div>
                <div>Queue: {i.totalQueue.toFixed(0)}</div>
                <div style={{ height: 6, background: '#f1f1f1', borderRadius: 3, marginTop: 8 }}>
                  <div style={{ width: Math.min(100, i.totalQueue * 5) + '%', height: '100%', background: '#3b82f6', borderRadius: 3 }} />
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      <div style={{ marginTop: 16 }}>
        <IntersectionVisualization 
          intersections={data?.intersections || []} 
          width={800} 
          height={400}
        />
      </div>
    </div>
  )
}


