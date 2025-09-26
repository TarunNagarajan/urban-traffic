import os
import sys
import numpy as np
import torch
import sumo_rl
import sumolib
from datetime import datetime
import json
import argparse
import signal

from agent import D3QNAgent
from config import SUMO_CONFIG, AGENT_CONFIG, TRAINING_CONFIG, REWARD_CONFIG, STUB_GRU_PREDICTION, SCHEDULER_CONFIG

# --- Constants based on environment inspection ---
NUM_PHASES = 2
NUM_LANES = 4

# Observation vector indices
PHASE_START, PHASE_END = 0, NUM_PHASES
MIN_GREEN_IDX = NUM_PHASES
DENSITY_START, DENSITY_END = MIN_GREEN_IDX + 1, MIN_GREEN_IDX + 1 + NUM_LANES
QUEUE_START, QUEUE_END = DENSITY_END, DENSITY_END + NUM_LANES

# Ensure SUMO_HOME is set
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

def get_neighboring_traffic_lights(net, ts_id):
    """
    Get neighboring traffic light IDs for a given traffic light.
    """
    try:
        node = net.getNode(ts_id)
        if node is None:
            return []
        
        incoming_edges = node.getIncoming()
        outgoing_edges = node.getOutgoing()
        connected_edges = list(incoming_edges) + list(outgoing_edges)
        
        neighboring_tls = set()
        for edge in connected_edges:
            from_node = edge.getFromNode()
            to_node = edge.getToNode()
            
            for other_node in [from_node, to_node]:
                if other_node and other_node.getID() != ts_id:
                    other_node_id = other_node.getID()
                    if other_node_id in [tls.getID() for tls in net.getTrafficLights()]:
                        neighboring_tls.add(other_node_id)
        return list(neighboring_tls)
    except Exception as e:
        print(f"Warning: Could not find neighbors for {ts_id}: {e}")
        return []

def compute_state(net, ts_id, obs, gru_stub_dim, max_neighbors, env):
    """
    Computes the state representation from the observation vector, including neighbor data and lane speeds.
    """
    local_obs = obs[ts_id]
    lane_queues = local_obs[QUEUE_START:QUEUE_END]
    
    lane_speeds = []
    for lane in env.traffic_signals[ts_id].lanes:
        lane_speeds.append(env.sumo.lane.getLastStepMeanSpeed(lane))
    
    max_lanes_for_padding = 8 
    if hasattr(env, 'max_lanes'):
        max_lanes_for_padding = env.max_lanes
    lane_speeds.extend([-1] * (max_lanes_for_padding - len(lane_speeds)))
    lane_speeds = np.array(lane_speeds)

    phase_one_hot = local_obs[PHASE_START:PHASE_END]
    
    state = np.concatenate([lane_queues, lane_speeds, phase_one_hot])

    neighbor_ids = get_neighboring_traffic_lights(net, ts_id)
    num_neighbors = 0
    for neighbor_id in neighbor_ids:
        if neighbor_id in obs:
            neighbor_obs = obs[neighbor_id]
            neighbor_phase = neighbor_obs[PHASE_START:PHASE_END]
            neighbor_queue = neighbor_obs[QUEUE_START:QUEUE_END]
            state = np.concatenate([state, neighbor_queue, neighbor_phase])
            num_neighbors += 1

    padding_size = (max_neighbors - num_neighbors) * (NUM_LANES + NUM_PHASES)
    if padding_size > 0:
        state = np.concatenate([state, np.zeros(padding_size)])

    predicted_inflow = np.random.rand(gru_stub_dim)
    state = np.concatenate([state, predicted_inflow])
    return state

def compute_reward(env, obs, prev_obs, REWARD_CONFIG, prev_arrived):
    """
    Computes a global reward and its components based on the chosen configuration.
    """
    reward_info = {}
    total_reward = 0

    if REWARD_CONFIG.get("use_pressure", False):
        pressure = 0
        for ts_id in obs.keys():
            pressure += env.traffic_signals[ts_id].get_pressure()
        reward_info['pressure'] = pressure * REWARD_CONFIG["pressure_weight"]
        total_reward += reward_info['pressure']

    if REWARD_CONFIG.get("use_throughput", False):
        arrived = env.sumo.simulation.getArrivedNumber()
        throughput = arrived - prev_arrived
        reward_info['throughput'] = throughput * REWARD_CONFIG["throughput_weight"]
        total_reward += reward_info['throughput']

    if REWARD_CONFIG.get("use_queue_length", False):
        total_queue = 0
        for ts_id in obs.keys():
            total_queue += np.sum(obs[ts_id][QUEUE_START:QUEUE_END])
        reward_info['queue_length'] = total_queue * REWARD_CONFIG["queue_length_weight"]
        total_reward += reward_info['queue_length']

    if REWARD_CONFIG.get("use_virtual_pedestrian_penalty", False):
        virtual_pedestrian_wait = 0
        for ts_id in obs.keys():
            virtual_pedestrian_wait += np.sum(obs[ts_id][QUEUE_START:QUEUE_END])
        reward_info['virtual_pedestrian_penalty'] = virtual_pedestrian_wait * REWARD_CONFIG["virtual_pedestrian_penalty"]
        total_reward += reward_info['virtual_pedestrian_penalty']

    if REWARD_CONFIG.get("use_fuel_consumption_penalty", False):
        total_fuel_consumption = 0
        for veh_id in env.sumo.vehicle.getIDList():
            total_fuel_consumption += env.sumo.vehicle.getFuelConsumption(veh_id)
        reward_info['fuel_consumption'] = total_fuel_consumption * REWARD_CONFIG["fuel_consumption_penalty"]
        total_reward += reward_info['fuel_consumption']

    if REWARD_CONFIG.get("use_jerk_penalty", False):
        jerk_penalty = 0
        for ts_id in obs.keys():
            if np.argmax(obs[ts_id][PHASE_START:PHASE_END]) != np.argmax(prev_obs[ts_id][PHASE_START:PHASE_END]):
                jerk_penalty += REWARD_CONFIG["jerk_penalty"]
        reward_info['jerk_penalty'] = jerk_penalty
        total_reward += jerk_penalty
    
    reward_info['total_reward'] = total_reward
    return reward_info

def save_checkpoint(episode, agent, log_dir):
    """
    Saves a training checkpoint.
    """
    checkpoint_path = os.path.join(log_dir, f"checkpoint_ep{episode}.pth")
    torch.save({
        'episode': episode,
        'model_state_dict': agent.qnetwork_local.state_dict(),
        'optimizer_state_dict': agent.optimizer.state_dict(),
        'epsilon': agent.epsilon,
    }, checkpoint_path)
    print(f"\nCheckpoint saved to {checkpoint_path}")

def train(REWARD_CONFIG, checkpoint_path=None, start_episode=1):
    """Main training loop."""
    TRAINING_CONFIG['episodes'] = 500 # Default total episodes
    
    log_dir = os.path.join(TRAINING_CONFIG["log_dir"], datetime.now().strftime('%Y%m%d-%H%M%S'))
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(os.path.dirname(TRAINING_CONFIG["model_save_path"]), exist_ok=True)

    env = sumo_rl.SumoEnvironment(
        net_file=SUMO_CONFIG["net_file"],
        route_file=SUMO_CONFIG["route_file"],
        out_csv_name=os.path.join(log_dir, "output.csv"),
        use_gui=SUMO_CONFIG["gui"],
        num_seconds=SUMO_CONFIG["num_seconds"],
        delta_time=SUMO_CONFIG["delta_time"],
        yellow_time=SUMO_CONFIG["yellow_time"],
        min_green=SUMO_CONFIG.get("min_green", 10),
        max_depart_delay=0
    )
    
    net = sumolib.net.readNet(env._net)

    initial_obs = env.reset()
    ts_ids = list(initial_obs.keys())
    
    max_neighbors = 0
    for ts_id in ts_ids:
        neighbors = get_neighboring_traffic_lights(net, ts_id)
        if len(neighbors) > max_neighbors:
            max_neighbors = len(neighbors)

    first_ts_id = ts_ids[0]
    state_size = compute_state(net, first_ts_id, initial_obs, STUB_GRU_PREDICTION["inflow_dimension"], max_neighbors, env).shape[0]
    AGENT_CONFIG["state_size"] = state_size

    agent = D3QNAgent(state_size=AGENT_CONFIG["state_size"], action_size=AGENT_CONFIG["action_size"])
    
    scheduler = None
    if SCHEDULER_CONFIG.get("use_lr_scheduler", False):
        scheduler = torch.optim.lr_scheduler.StepLR(
            agent.optimizer,
            step_size=SCHEDULER_CONFIG["step_size"],
            gamma=SCHEDULER_CONFIG["gamma"]
        )

    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path)
        agent.load_checkpoint(checkpoint)
        print(f"Resumed training from checkpoint: {checkpoint_path}")

    best_reward = -np.inf

    def signal_handler(sig, frame):
        print('\nCaught Ctrl+C. Saving model and exiting...')
        agent.save(TRAINING_CONFIG["model_save_path"])
        env.close()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    for episode in range(start_episode, TRAINING_CONFIG["episodes"] + 1):
        obs = env.reset()
        
        episode_rewards = {}

        done = {"__all__": False}
        prev_obs = obs
        prev_arrived = 0
        
        while not done["__all__"]:
            action_dict = {}
            states = {}
            meta_actions = {}
            
            for ts_id in ts_ids:
                if env.traffic_signals[ts_id].time_since_last_phase_change < env.traffic_signals[ts_id].min_green:
                    action_mask = np.array([1, 0])
                else:
                    action_mask = np.array([1, 1])

                states[ts_id] = compute_state(net, ts_id, obs, STUB_GRU_PREDICTION["inflow_dimension"], max_neighbors, env)
                meta_actions[ts_id] = agent.act(states[ts_id], agent.epsilon, action_mask)
                
                current_phase = np.argmax(obs[ts_id][PHASE_START:PHASE_END])
                if meta_actions[ts_id] == 0:
                    action_dict[ts_id] = current_phase
                else:
                    action_dict[ts_id] = (current_phase + 1) % NUM_PHASES

            next_obs, _, done, _ = env.step(action_dict)

            reward_info = compute_reward(env, next_obs, prev_obs, REWARD_CONFIG, prev_arrived)
            reward = reward_info['total_reward']
            prev_arrived = env.sumo.simulation.getArrivedNumber()
            
            for ts_id in ts_ids:
                next_state = compute_state(net, ts_id, next_obs, STUB_GRU_PREDICTION["inflow_dimension"], max_neighbors, env)
                agent.step(states[ts_id], meta_actions[ts_id], reward, next_state, done["__all__"])

            prev_obs = obs
            obs = next_obs
            
            for key, value in reward_info.items():
                episode_rewards[key] = episode_rewards.get(key, 0) + value

        agent.epsilon = max(AGENT_CONFIG["epsilon_end"], AGENT_CONFIG["epsilon_decay"] * agent.epsilon)
        
        if scheduler:
            scheduler.step()

        reward_summary = " | ".join([f"{key}: {value:.2f}" for key, value in episode_rewards.items()])
        current_lr = agent.optimizer.param_groups[0]['lr']
        print(f"Episode {episode}/{TRAINING_CONFIG['episodes']} | {reward_summary} | Epsilon: {agent.epsilon:.4f} | LR: {current_lr:.6f}")

        if episode_rewards['total_reward'] > best_reward:
            best_reward = episode_rewards['total_reward']
            agent.save(TRAINING_CONFIG["model_save_path"])
            print(f"New best model saved with reward: {best_reward:.2f}")

        if episode % TRAINING_CONFIG["save_checkpoint_every"] == 0:
            save_checkpoint(episode, agent, log_dir)

    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a D3QN agent for traffic light control.')
    parser.add_argument('--config', type=str, default='default', help='Name of the reward configuration file to use (without .json extension).')
    parser.add_argument('--checkpoint_path', type=str, help='Path to a model checkpoint to resume training.')
    parser.add_argument('--start_episode', type=int, default=1, help='Episode to start training from.')
    parser.add_argument('--episodes', type=int, help='Set the total number of episodes to train for.')
    args = parser.parse_args()

    if args.episodes:
        TRAINING_CONFIG['episodes'] = args.episodes

    config_path = os.path.join('configs', f'{args.config}.json')
    if not os.path.exists(config_path):
        sys.exit(f"Error: Configuration file not found at {config_path}")

    with open(config_path, 'r') as f:
        REWARD_CONFIG = json.load(f)
    
    print(f"--- Using reward configuration: {args.config} ---")
    train(REWARD_CONFIG, checkpoint_path=args.checkpoint_path, start_episode=args.start_episode)