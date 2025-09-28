
import os
os.environ['PROJ_NETWORK'] = 'OFF'
import sys
import numpy as np
import torch
import traci
import json
import argparse
from sumo_rl import SumoEnvironment
from agent import D3QNAgent
from config import AGENT_CONFIG

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

def custom_reward_function(traffic_signal):
    # The 'traffic_signal' argument is a sumo_rl.environment.traffic_signal.TrafficSignal object
    ts_id = traffic_signal.id

    # Weights for each component of the reward function
    W_QUEUE = -0.5
    W_WAITING = -0.5
    W_FLOW = 1.0
    W_JERK = -0.1
    W_SWITCH = -0.5

    # Get controlled lanes
    lanes = traci.trafficlight.getControlledLanes(ts_id)
    
    # 1. Queue Length
    queue_length = 0
    for lane in lanes:
        queue_length += traci.lane.getLastStepHaltingNumber(lane)

    # 2. Waiting Time
    waiting_time = 0
    for lane in lanes:
        waiting_time += traci.lane.getWaitingTime(lane)

    # 3. Flow
    flow = 0
    for lane in lanes:
        flow += traci.lane.getLastStepVehicleNumber(lane)

    # 4. Jerk
    jerk = 0
    vehicles = traci.vehicle.getIDList()
    for vehicle_id in vehicles:
        if traci.vehicle.getLaneID(vehicle_id) in lanes:
            accel = traci.vehicle.getAcceleration(vehicle_id)
            # Approximate jerk as change in acceleration
            # This requires storing previous acceleration, for simplicity we use accel^2
            jerk += accel**2

    # 5. Frequent Switching
    # This needs to be handled in the environment or agent logic
    # For now, we'll add a placeholder. A simple way is to penalize any action that changes the phase.
    # A more sophisticated approach would be to check if the new phase is different from the old one.
    # This is handled by the environment's logic, so we can't directly implement it here.
    # We will assume the environment provides a penalty for switching.
    switch_penalty = 0 # Placeholder

    reward = (
        W_QUEUE * queue_length +
        W_WAITING * waiting_time +
        W_FLOW * flow +
        W_JERK * jerk +
        W_SWITCH * switch_penalty
    )

    return reward

class CommunicatingSumoEnvironment(SumoEnvironment):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.neighbor_radius = 1000
        self.neighbors = None # Initialize as None

    def _get_neighbors(self):
        # This method now needs to be called only when self.sumo is valid
        neighbors = {ts_id: [] for ts_id in self.ts_ids}
        positions = {ts_id: self.sumo.junction.getPosition(ts_id) for ts_id in self.ts_ids}

        for ts_id_1 in self.ts_ids:
            for ts_id_2 in self.ts_ids:
                if ts_id_1 != ts_id_2:
                    dist = np.linalg.norm(np.array(positions[ts_id_1]) - np.array(positions[ts_id_2]))
                    if dist < self.neighbor_radius:
                        neighbors[ts_id_1].append(ts_id_2)
        return neighbors

    def _compute_observations(self):
        # Check if neighbors have been computed. If not, compute them.
        if self.neighbors is None:
            # This is the first time observations are computed after a reset,
            # so self.sumo should be available.
            self.neighbors = self._get_neighbors()

        observations = super()._compute_observations()
        communicating_observations = {}
        for ts_id in self.ts_ids:
            # Build a list of all observation arrays to concatenate
            all_obs_to_concat = [observations[ts_id]]
            for neighbor_id in self.neighbors.get(ts_id, []):
                if neighbor_id in observations:
                    all_obs_to_concat.append(observations[neighbor_id])
            
            # Concatenate all observations
            communicating_observations[ts_id] = np.concatenate(all_obs_to_concat)
        return communicating_observations

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train D3QN agents for traffic signal control.')
    parser.add_argument('--resume', type=str, help='Path to the checkpoint file to resume training from.')
    args = parser.parse_args()

    env = CommunicatingSumoEnvironment(net_file='Bhubaneswar.net.xml',
                                     route_file='Bhubaneswar.rou.xml',
                                     out_csv_name='outputs/bhubaneswar_d3qn',
                                     use_gui=False,
                                     num_seconds=10800,  # 3 hours
                                     yellow_time=3,
                                     min_green=5,
                                     max_green=60,
                                     reward_fn=custom_reward_function)

    # Initialize the environment to get observation and action spaces
    initial_states = env.reset()
    ts_ids = list(initial_states.keys())
    action_size = env.action_space.n

    # Create a D3QN agent for each traffic light with its specific state size
    agents = {ts_id: D3QNAgent(initial_states[ts_id].shape[0], action_size) for ts_id in ts_ids}

    start_episode = 0
    if args.resume:
        with open(args.resume, 'r') as f:
            checkpoint_info = json.load(f)
        start_episode = checkpoint_info['episode']
        for ts_id in env.ts_ids:
            model_path = checkpoint_info['models'][ts_id]
            if os.path.exists(model_path):
                state_dict = torch.load(model_path)
                agents[ts_id].qnetwork_local.load_state_dict(state_dict)
                agents[ts_id].qnetwork_target.load_state_dict(state_dict)
                agents[ts_id].epsilon = checkpoint_info['epsilons'][ts_id]
            else:
                print(f"Warning: Checkpoint model file not found for agent {ts_id}: {model_path}")


    best_reward = -np.inf
    epsilon_start = AGENT_CONFIG['epsilon_start']
    epsilon_end = AGENT_CONFIG['epsilon_end']
    epsilon_decay = AGENT_CONFIG['epsilon_decay']

    # Training loop
    for e in range(start_episode, 100): # Number of episodes
        states = env.reset()
        done = {'__all__': False}
        total_reward = 0

        while not done['__all__']:
            actions = {ts_id: agents[ts_id].act(states[ts_id], eps=agents[ts_id].epsilon) for ts_id in env.ts_ids}
            next_states, rewards, done, _ = env.step(action=actions)

            for ts_id in env.ts_ids:
                agents[ts_id].step(states[ts_id], actions[ts_id], rewards[ts_id], next_states[ts_id], done[ts_id])
                total_reward += rewards[ts_id]

            states = next_states

            if done['__all__']:
                print("Episode: {}/{}, Total Reward: {}, Epsilon: {:.2}".format(e, 100, total_reward, agents[list(env.ts_ids)[0]].epsilon))
                
                # Decay epsilon for all agents
                for agent in agents.values():
                    agent.epsilon = max(epsilon_end, epsilon_decay * agent.epsilon)

                # Save checkpoint
                checkpoint_data = {
                    'episode': e + 1,
                    'models': {ts_id: f'models/{ts_id}_d3qn_ep{e}.pth' for ts_id in env.ts_ids},
                    'epsilons': {ts_id: agents[ts_id].epsilon for ts_id in env.ts_ids}
                }
                for ts_id, agent in agents.items():
                    agent.save(checkpoint_data['models'][ts_id])
                
                os.makedirs('checkpoints', exist_ok=True)
                with open(f'checkpoints/d3qn_checkpoint_ep{e}.json', 'w') as f:
                    json.dump(checkpoint_data, f)

                # Save best model
                if total_reward > best_reward:
                    best_reward = total_reward
                    for ts_id, agent in agents.items():
                        agent.save(f'models/{ts_id}_d3qn_best.pth')
                    print("New best model saved.")

    env.close()
