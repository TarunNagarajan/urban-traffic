import asyncio
import os
import sumo_rl
import sumolib
import torch
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from threading import Thread, Event
from typing import Dict, Any, Optional

from agent import D3QNAgent
from config import SUMO_CONFIG, AGENT_CONFIG, TRAINING_CONFIG, STUB_GRU_PREDICTION
from train import compute_state, get_neighboring_traffic_lights, NUM_PHASES, PHASE_START, PHASE_END, QUEUE_START, QUEUE_END


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SimulationRunner:
    def __init__(self) -> None:
        self.env = None
        self.net = None
        self.agent: Optional[D3QNAgent] = None
        self.max_neighbors = 0
        self.ts_ids = []
        self.thread: Optional[Thread] = None
        self.stop_event = Event()
        self.is_running = False

    def _build_env(self) -> None:
        self.env = sumo_rl.SumoEnvironment(
            net_file='C:/Users/ultim/anaconda3/envs/metaworld-cpu/lib/site-packages/sumo_rl/nets/4x4-Lucas/4x4.net.xml',
            route_file='C:/Users/ultim/anaconda3/envs/metaworld-cpu/lib/site-packages/sumo_rl/nets/4x4-Lucas/4x4c1.rou.xml',
            out_csv_name=None,
            use_gui=SUMO_CONFIG["gui"],
            num_seconds=SUMO_CONFIG["num_seconds"],
            delta_time=SUMO_CONFIG["delta_time"],
            yellow_time=SUMO_CONFIG["yellow_time"],
            max_depart_delay=0,
        )
        self.net = sumolib.net.readNet(self.env._net)

    def _load_agent(self) -> None:
        initial_obs = self.env.reset()
        self.ts_ids = list(initial_obs.keys())

        self.max_neighbors = 0
        for ts_id in self.ts_ids:
            neighbors = get_neighboring_traffic_lights(self.net, ts_id)
            if len(neighbors) > self.max_neighbors:
                self.max_neighbors = len(neighbors)

        first_ts_id = self.ts_ids[0]
        state_size = compute_state(self.net, first_ts_id, initial_obs, STUB_GRU_PREDICTION["inflow_dimension"], self.max_neighbors).shape[0]

        agent = D3QNAgent(state_size=state_size, action_size=AGENT_CONFIG["action_size"])
        model_path = TRAINING_CONFIG["model_save_path"]
        if os.path.exists(model_path):
            agent.qnetwork_local.load_state_dict(torch.load(model_path, map_location=agent.device))
            self.agent = agent
        else:
            self.agent = None

    def start(self) -> None:
        if self.is_running:
            return
        self.stop_event.clear()
        self._build_env()
        self._load_agent()
        self.is_running = True

    def stop(self) -> None:
        self.stop_event.set()
        if self.env is not None:
            try:
                self.env.close()
            except Exception:
                pass
        self.is_running = False

    def step(self) -> Dict[str, Any]:
        if not self.is_running or self.env is None:
            return {"status": "idle"}

        obs = self.env._obs
        if obs is None:
            obs = self.env.reset()

        action_dict: Dict[str, int] = {}
        for ts_id in self.ts_ids:
            state = compute_state(self.net, ts_id, obs, STUB_GRU_PREDICTION["inflow_dimension"], self.max_neighbors)
            current_phase = int(np.argmax(obs[ts_id][PHASE_START:PHASE_END]))
            if self.agent is None:
                action = current_phase  # baseline: hold
            else:
                meta_action = int(self.agent.act(state, eps=0.0))
                action = current_phase if meta_action == 0 else (current_phase + 1) % NUM_PHASES
            action_dict[ts_id] = action

        next_obs, _, done, _ = self.env.step(action_dict)

        payload: Dict[str, Any] = {"done": bool(done.get("__all__", False)), "intersections": []}
        for ts_id in self.ts_ids:
            phase = int(np.argmax(next_obs[ts_id][PHASE_START:PHASE_END]))
            total_queue = float(np.sum(next_obs[ts_id][QUEUE_START:QUEUE_END]))
            payload["intersections"].append({
                "id": ts_id,
                "phase": phase,
                "totalQueue": total_queue,
            })

        return payload


runner = SimulationRunner()


@app.post("/start")
def start_sim() -> Dict[str, Any]:
    runner.start()
    return {"status": "started", "gui": SUMO_CONFIG["gui"]}


@app.post("/stop")
def stop_sim() -> Dict[str, Any]:
    runner.stop()
    return {"status": "stopped"}


@app.websocket("/ws")
async def ws_stream(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        while True:
            if not runner.is_running:
                await asyncio.sleep(0.2)
                continue
            data = runner.step()
            await websocket.send_json(data)
            if data.get("done"):
                runner.stop()
            await asyncio.sleep(SUMO_CONFIG["delta_time"])  # seconds between actions
    except WebSocketDisconnect:
        return


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


