
import React, { useState, useEffect, useRef } from 'react';
import { Canvas } from '@react-three/fiber';
import { MapControls, Line } from '@react-three/drei';
import * as THREE from 'three';

const WEBSOCKET_URL = 'ws://localhost:8080';

// --- Helper Components ---

const Road = ({ shape }) => {
    // Convert [x, y] points to [x, 0, y] for 3D space (x-z plane)
    const points = shape.map(p => new THREE.Vector3(p[0], 0, p[1]));
    return <Line points={points} color="grey" lineWidth={2} />;
};

const Vehicle = ({ data }) => {
    const meshRef = useRef();
    // Position vehicle in 3D space (x-z plane)
    const position = [data.x, 0.5, data.y]; // y=0.5 to be above the road

    useEffect(() => {
        if (meshRef.current) {
            // SUMO angle is clockwise from North, convert to radians for Three.js
            const angleRad = (data.angle * Math.PI) / 180;
            // We subtract the angle from PI/2 to align with the 3D coordinate system
            meshRef.current.rotation.y = Math.PI / 2 - angleRad;
        }
    }, [data.angle]);

    return (
        <mesh ref={meshRef} position={position}>
            <boxGeometry args={[4, 1, 2]} />
            <meshStandardMaterial color="red" />
        </mesh>
    );
};

// --- Main Component ---

const LiveTraffic = () => {
    const [mapData, setMapData] = useState([]);
    const [vehicles, setVehicles] = useState({});

    // Load static map data
    useEffect(() => {
        fetch('/map_data.json')
            .then(res => res.json())
            .then(data => {
                console.log("Map data loaded:", data);
                setMapData(data);
            })
            .catch(err => console.error("Failed to load map data:", err));
    }, []);

    // Connect to WebSocket for live data
    useEffect(() => {
        const ws = new WebSocket(WEBSOCKET_URL);

        ws.onopen = () => {
            console.log('Connected to WebSocket server');
        };

        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            if (data.vehicles) {
                // Update vehicle data using a key-based object for efficiency
                setVehicles(prevVehicles => {
                    const updated = {};
                    data.vehicles.forEach(v => {
                        updated[v.id] = v;
                    });
                    return updated;
                });
            }
        };

        ws.onclose = () => {
            console.log('Disconnected from WebSocket server');
        };

        ws.onerror = (error) => {
            console.error('WebSocket error:', error);
        };

        // Cleanup on component unmount
        return () => {
            ws.close();
        };
    }, []);

    return (
        <Canvas camera={{ position: [0, 150, 0], fov: 50 }}>
            <ambientLight intensity={0.8} />
            <directionalLight position={[10, 10, 5]} intensity={1} />
            <MapControls />

            {/* Render the road network */}
            {mapData.map(road => (
                <Road key={road.id} shape={road.shape} />
            ))}

            {/* Render the vehicles */}
            {Object.values(vehicles).map(vehicle => (
                <Vehicle key={vehicle.id} data={vehicle} />
            ))}

            {/* Simple ground plane */}
            <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -0.1, 0]}>
                <planeGeometry args={[1000, 1000]} />
                <meshStandardMaterial color="#444" />
            </mesh>
        </Canvas>
    );
};

export default LiveTraffic;
