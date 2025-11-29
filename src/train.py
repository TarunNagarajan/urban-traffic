
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import sumo_rl
import sumolib
from datetime import datetime
import json
import argparse
import signal
import logging
import pandas as pd
from pathlib import Path
import importlib.resources as pkg_resources
from collections import defaultdict, deque
import time

from agent import D3QNAgent
from config import SUMO_CONFIG, AGENT_CONFIG, TRAINING_CONFIG, REWARD_CONFIG, STUB_GRU_PREDICTION

# --- Constants based on environment inspection ---
NUM_PHASES = 2
NUM_LANES = 4

# Observation vector indices
PHASE_START, PHASE_END = 0, NUM_PHASES
MIN_GREEN_IDX = NUM_PHASES
DENSITY_START, DENSITY_END = MIN_GREEN_IDX + 1, MIN_GREEN_IDX + 1 + NUM_LANES
QUEUE_START, QUEUE_END = DENSITY_END, DENSITY_END + NUM_LANES

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
        logger.warning(f"Could not find neighbors for {ts_id}: {e}")
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

def compute_reward(env, obs, prev_obs, REWARD_CONFIG, prev_arrived, step_info=None):
    """
    Computes a global reward and its components based on the chosen configuration.
    Enhanced with step_info for more detailed metrics.
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

    # Additional reward components for better traffic management
    if REWARD_CONFIG.get("use_waiting_time", False):
        waiting_time_penalty = 0
        for veh_id in env.sumo.vehicle.getIDList():
            waiting_time_penalty += env.sumo.vehicle.getWaitingTime(veh_id)
        reward_info['waiting_time'] = waiting_time_penalty * REWARD_CONFIG.get("waiting_time_weight", -0.01)
        total_reward += reward_info['waiting_time']

    if REWARD_CONFIG.get("use_speed_efficiency", False):
        avg_speed = 0
        vehicle_count = len(env.sumo.vehicle.getIDList())
        if vehicle_count > 0:
            for veh_id in env.sumo.vehicle.getIDList():
                avg_speed += env.sumo.vehicle.getSpeed(veh_id)
            avg_speed /= vehicle_count
        reward_info['speed_efficiency'] = avg_speed * REWARD_CONFIG.get("speed_efficiency_weight", 0.01)
        total_reward += reward_info['speed_efficiency']

    reward_info['total_reward'] = total_reward
    return reward_info

class MultiAgentSystem:
    """
    A system to manage multiple D3QN agents for different traffic intersections.
    """
    def __init__(self, state_size, action_size, ts_ids, config):
        self.agents = {}
        self.ts_ids = ts_ids
        self.config = config

        # Create individual agents for each intersection
        for ts_id in ts_ids:
            agent = D3QNAgent(state_size=state_size, action_size=action_size)

            # Apply individual agent hyperparameters based on config
            if hasattr(config, 'agent_specific_configs') and ts_id in config.agent_specific_configs:
                for attr, value in config.agent_specific_configs[ts_id].items():
                    setattr(agent, attr, value)

            self.agents[ts_id] = agent

    def step(self, ts_id, state, action, reward, next_state, done):
        """Update a specific agent"""
        return self.agents[ts_id].step(state, action, reward, next_state, done)

    def act(self, ts_id, state, epsilon, action_mask=None, temperature=0.0):
        """Get action from a specific agent"""
        return self.agents[ts_id].act(state, epsilon, action_mask, temperature)

    def update_target_networks(self):
        """Update target networks for all agents"""
        for agent in self.agents.values():
            # Copy local network to target network
            agent.qnetwork_target.load_state_dict(agent.qnetwork_local.state_dict())

    def save_agent(self, ts_id, path):
        """Save a specific agent"""
        return self.agents[ts_id].save(path)

    def save_all_agents(self, base_path):
        """Save all agents to individual files"""
        for ts_id, agent in self.agents.items():
            path = f"{base_path}_agent_{ts_id}.pth"
            agent.save(path)

    def get_epsilon(self, ts_id):
        """Get epsilon for specific agent"""
        return self.agents[ts_id].epsilon

    def set_epsilon(self, ts_id, epsilon):
        """Set epsilon for specific agent"""
        self.agents[ts_id].epsilon = epsilon

class TrainingLogger:
    """
    Handles logging of training metrics and performance.
    """
    def __init__(self, log_dir):
        self.log_dir = log_dir
        self.episode_metrics = []
        self.training_stats = defaultdict(list)

        # Create log directories
        os.makedirs(os.path.join(log_dir, "metrics"), exist_ok=True)
        os.makedirs(os.path.join(log_dir, "checkpoints"), exist_ok=True)

    def log_episode(self, episode, metrics):
        """Log metrics for a single episode"""
        episode_data = {
            'episode': episode,
            'timestamp': datetime.now().isoformat(),
            **metrics
        }
        self.episode_metrics.append(episode_data)

    def save_episode_metrics(self, episode):
        """Save episode metrics to CSV"""
        if self.episode_metrics:
            df = pd.DataFrame([self.episode_metrics[-1]])  # Save just the most recent
            metrics_file = os.path.join(self.log_dir, "metrics", f"episode_{episode}_metrics.csv")
            df.to_csv(metrics_file, index=False)

    def save_cumulative_metrics(self):
        """Save all metrics to a single file periodically"""
        if self.episode_metrics:
            df = pd.DataFrame(self.episode_metrics)
            cumulative_file = os.path.join(self.log_dir, "cumulative_metrics.csv")
            df.to_csv(cumulative_file, index=False)

    def log_training_step(self, step_data):
        """Log data from individual training steps"""
        for key, value in step_data.items():
            self.training_stats[key].append(value)

def save_checkpoint(episode, multi_agent_system, log_dir, additional_info=None):
    """
    Saves a training checkpoint for all agents.
    """
    checkpoint_data = {
        'episode': episode,
        'timestamp': datetime.now().isoformat(),
    }

    # Save state for each agent
    for ts_id, agent in multi_agent_system.agents.items():
        checkpoint_data[f'model_state_dict_{ts_id}'] = agent.qnetwork_local.state_dict()
        checkpoint_data[f'optimizer_state_dict_{ts_id}'] = agent.optimizer.state_dict()
        checkpoint_data[f'epsilon_{ts_id}'] = agent.epsilon

    # Add any additional information
    if additional_info:
        checkpoint_data.update(additional_info)

    checkpoint_path = os.path.join(log_dir, "checkpoints", f"checkpoint_ep{episode}.pth")
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    torch.save(checkpoint_data, checkpoint_path)
    logger.info(f"Checkpoint saved to {checkpoint_path}")
    return checkpoint_path

def validate_agent(env, multi_agent_system, net, ts_ids, max_neighbors, num_steps=100):
    """
    Validate the current agent performance without training
    """
    env.reset()
    obs = env.reset()
    total_reward = 0
    steps = 0
    done = {"__all__": False}

    # Temporarily set epsilon to 0 to get deterministic actions
    original_epsilons = {}
    for ts_id in ts_ids:
        original_epsilons[ts_id] = multi_agent_system.get_epsilon(ts_id)
        multi_agent_system.set_epsilon(ts_id, 0.0)  # Deterministic actions

    while not done["__all__"] and steps < num_steps:
        action_dict = {}

        for ts_id in ts_ids:
            if env.traffic_signals[ts_id].time_since_last_phase_change < env.traffic_signals[ts_id].min_green:
                action_mask = np.array([1, 0])
            else:
                action_mask = np.array([1, 1])

            state = compute_state(net, ts_id, obs, STUB_GRU_PREDICTION["inflow_dimension"], max_neighbors, env)
            meta_action = multi_agent_system.act(ts_id, state, 0.0, action_mask)  # No exploration

            current_phase = np.argmax(obs[ts_id][PHASE_START:PHASE_END])
            if meta_action == 0:
                action_dict[ts_id] = current_phase
            else:
                action_dict[ts_id] = (current_phase + 1) % NUM_PHASES

        next_obs, rewards, done, _ = env.step(action_dict)

        # Calculate validation reward (using a simplified approach)
        for ts_id in ts_ids:
            total_reward += np.sum(rewards.get(ts_id, 0))

        obs = next_obs
        steps += 1

    # Restore original epsilons
    for ts_id, epsilon in original_epsilons.items():
        multi_agent_system.set_epsilon(ts_id, epsilon)

    avg_reward = total_reward / max(steps, 1)
    logger.info(f"Validation - Average Reward: {avg_reward:.2f}, Steps: {steps}")
    return avg_reward

def train(REWARD_CONFIG, checkpoint_path=None, start_episode=1, network_topology='4x4-Lucas',
          validate_freq=None, seed=None):
    """
    Enhanced training loop with multi-agent support, validation, and comprehensive logging.
    """
    # Set random seed for reproducibility if provided
    if seed:
        np.random.seed(seed)
        torch.manual_seed(seed)
        logger.info(f"Set random seed to {seed}")

    # Update episodes from config
    episodes = TRAINING_CONFIG.get('episodes', 480)

    log_dir = os.path.join(TRAINING_CONFIG["log_dir"], datetime.now().strftime('%Y%m%d-%H%M%S'))
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(os.path.dirname(TRAINING_CONFIG["model_save_path"]), exist_ok=True)

    # Handle network topology configuration
    try:
        base_path = pkg_resources.files('sumo_rl') / 'nets' / network_topology
        net_file = str(base_path / '4x4.net.xml')
        route_file = str(base_path / '4x4c1.rou.xml')

        # Check if the network topology files exist, fallback to default if not
        if not os.path.exists(net_file):
            logger.warning(f"Network topology {network_topology} not found, using default 4x4-Lucas")
            base_path = pkg_resources.files('sumo_rl') / 'nets' / '4x4-Lucas'
            net_file = str(base_path / '4x4.net.xml')
            route_file = str(base_path / '4x4c1.rou.xml')
    except Exception as e:
        logger.warning(f"Could not load network topology {network_topology}, using default: {e}")
        base_path = pkg_resources.files('sumo_rl') / 'nets' / '4x4-Lucas'
        net_file = str(base_path / '4x4.net.xml')
        route_file = str(base_path / '4x4c1.rou.xml')

    env = sumo_rl.SumoEnvironment(
        net_file=net_file,
        route_file=route_file,
        out_csv_name=os.path.join(log_dir, "output.csv"),
        use_gui=SUMO_CONFIG.get("gui", False),
        num_seconds=SUMO_CONFIG.get("num_seconds", 900),
        delta_time=SUMO_CONFIG.get("delta_time", 5),
        yellow_time=SUMO_CONFIG.get("yellow_time", 2),
        min_green=SUMO_CONFIG.get("min_green", 10),
        max_depart_delay=0
    )

    net = sumolib.net.readNet(env._net)

    initial_obs = env.reset()
    ts_ids = list(initial_obs.keys())

    # Calculate max neighbors for state computation
    max_neighbors = 0
    for ts_id in ts_ids:
        neighbors = get_neighboring_traffic_lights(net, ts_id)
        if len(neighbors) > max_neighbors:
            max_neighbors = len(neighbors)

    # Compute state size using first intersection
    first_ts_id = ts_ids[0]
    state_size = compute_state(net, first_ts_id, initial_obs,
                              STUB_GRU_PREDICTION["inflow_dimension"], max_neighbors, env).shape[0]

    logger.info(f"Environment initialized: {len(ts_ids)} intersections, state_size: {state_size}")

    # Initialize multi-agent system
    action_size = 2  # HOLD or SWITCH
    multi_agent_system = MultiAgentSystem(
        state_size=state_size,
        action_size=action_size,
        ts_ids=ts_ids,
        config=type('Config', (), {'agent_specific_configs': {}})  # Empty config for now
    )

    # Load checkpoint if provided
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)

        # Load models for each agent
        for ts_id in ts_ids:
            if f'model_state_dict_{ts_id}' in checkpoint:
                multi_agent_system.agents[ts_id].qnetwork_local.load_state_dict(
                    checkpoint[f'model_state_dict_{ts_id}']
                )
                multi_agent_system.agents[ts_id].qnetwork_target.load_state_dict(
                    checkpoint[f'model_state_dict_{ts_id}']
                )
                # Load optimizer if present
                if f'optimizer_state_dict_{ts_id}' in checkpoint:
                    multi_agent_system.agents[ts_id].optimizer.load_state_dict(
                        checkpoint[f'optimizer_state_dict_{ts_id}']
                    )
                # Load epsilon if present
                if f'epsilon_{ts_id}' in checkpoint:
                    multi_agent_system.agents[ts_id].epsilon = checkpoint[f'epsilon_{ts_id}']

        logger.info(f"Resumed training from checkpoint: {checkpoint_path}")

    # Initialize logger
    logger_instance = TrainingLogger(log_dir)
    best_reward = float('-inf')

    def signal_handler(sig, frame):
        logger.info('\nCaught Ctrl+C. Saving model and exiting...')
        multi_agent_system.save_all_agents(TRAINING_CONFIG["model_save_path"])
        env.close()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    logger.info(f"Starting training for {episodes} episodes...")
    start_time = time.time()

    for episode in range(start_episode, episodes + 1):
        obs = env.reset()

        # Initialize metrics for this episode
        episode_rewards = defaultdict(float)
        episode_steps = 0
        total_waiting_time = 0
        total_queue_length = 0

        done = {"__all__": False}
        prev_obs = obs
        prev_arrived = 0
        step_count = 0

        episode_start_time = time.time()

        while not done["__all__"]:
            action_dict = {}
            states = {}
            meta_actions = {}

            # Collect actions from all agents
            for ts_id in ts_ids:
                # Check if minimum green time constraint is met
                if env.traffic_signals[ts_id].time_since_last_phase_change < env.traffic_signals[ts_id].min_green:
                    action_mask = np.array([1, 0])  # Only allow HOLD
                else:
                    action_mask = np.array([1, 1])  # Allow both HOLD and SWITCH

                states[ts_id] = compute_state(net, ts_id, obs, STUB_GRU_PREDICTION["inflow_dimension"],
                                            max_neighbors, env)

                # Use epsilon-greedy with current agent's epsilon
                epsilon = multi_agent_system.get_epsilon(ts_id)
                meta_actions[ts_id] = multi_agent_system.act(ts_id, states[ts_id], epsilon, action_mask)

                # Translate meta-action to environment action
                current_phase = np.argmax(obs[ts_id][PHASE_START:PHASE_END])
                if meta_actions[ts_id] == 0:
                    action_dict[ts_id] = current_phase
                else:
                    action_dict[ts_id] = (current_phase + 1) % NUM_PHASES

            # Execute actions in environment
            next_obs, _, done, _ = env.step(action_dict)

            # Calculate rewards for all agents
            reward_info = compute_reward(env, next_obs, prev_obs, REWARD_CONFIG, prev_arrived)
            reward = reward_info['total_reward']
            prev_arrived = env.sumo.simulation.getArrivedNumber()

            # Update each agent with the global reward (can be modified for individual rewards)
            for ts_id in ts_ids:
                next_state = compute_state(net, ts_id, next_obs, STUB_GRU_PREDICTION["inflow_dimension"],
                                         max_neighbors, env)
                multi_agent_system.step(ts_id, states[ts_id], meta_actions[ts_id], reward,
                                      next_state, done["__all__"])

            # Update episode metrics
            for key, value in reward_info.items():
                episode_rewards[key] += value

            # Calculate additional metrics
            current_total_queue = sum(np.sum(obs[ts_id][QUEUE_START:QUEUE_END]) for ts_id in ts_ids)
            total_queue_length += current_total_queue

            # Count waiting vehicles for this step
            waiting_count = 0
            for veh_id in env.sumo.vehicle.getIDList():
                if env.sumo.vehicle.getWaitingTime(veh_id) > 0:
                    waiting_count += 1
            total_waiting_time += waiting_count

            prev_obs = obs
            obs = next_obs
            step_count += 1

        # Update agent epsilons (epsilon decay)
        for ts_id in ts_ids:
            agent = multi_agent_system.agents[ts_id]
            agent.epsilon = max(AGENT_CONFIG["epsilon_end"],
                               AGENT_CONFIG["epsilon_decay"] * agent.epsilon)

        # Calculate episode metrics
        avg_queue_length = total_queue_length / max(step_count, 1)
        avg_waiting_time = total_waiting_time / max(step_count, 1)

        episode_duration = time.time() - episode_start_time

        # Log episode metrics
        episode_metrics = {
            **episode_rewards,
            'steps': step_count,
            'avg_queue_length': avg_queue_length,
            'avg_waiting_time': avg_waiting_time,
            'episode_duration': episode_duration,
            'epsilon': np.mean([multi_agent_system.get_epsilon(ts_id) for ts_id in ts_ids])
        }

        logger_instance.log_episode(episode, episode_metrics)
        logger_instance.save_episode_metrics(episode)

        # Log to console
        reward_summary = " | ".join([f"{key}: {value:.2f}" for key, value in episode_metrics.items()
                                   if key in ['total_reward', 'queue_length', 'pressure', 'throughput']])
        logger.info(f"Episode {episode}/{episodes} | {reward_summary} | "
                   f"Steps: {step_count} | Duration: {episode_duration:.2f}s | "
                   f"Epsilon: {episode_metrics['epsilon']:.4f}")

        # Update best model if this episode had better reward
        current_total_reward = episode_metrics.get('total_reward', 0)
        if current_total_reward > best_reward:
            best_reward = current_total_reward
            best_model_path = TRAINING_CONFIG["model_save_path"]

            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(best_model_path), exist_ok=True)

            # Save all agents
            multi_agent_system.save_all_agents(best_model_path.replace('.pth', ''))
            logger.info(f"New best model saved with reward: {best_reward:.2f}")

        # Save checkpoint periodically
        if episode % TRAINING_CONFIG.get("save_checkpoint_every", 10) == 0:
            additional_info = {
                'best_reward_so_far': best_reward,
                'total_training_time': time.time() - start_time,
                'episode_metrics': episode_metrics
            }
            save_checkpoint(episode, multi_agent_system, log_dir, additional_info)

        # Perform validation periodically if enabled
        if validate_freq and episode % validate_freq == 0:
            logger.info(f"Performing validation at episode {episode}")
            val_reward = validate_agent(env, multi_agent_system, net, ts_ids, max_neighbors)
            episode_metrics['validation_reward'] = val_reward

    # Final cleanup
    total_training_time = time.time() - start_time
    logger.info(f"Training completed in {total_training_time:.2f} seconds")

    # Save final models
    multi_agent_system.save_all_agents(TRAINING_CONFIG["model_save_path"].replace('.pth', '_final'))

    # Save cumulative metrics
    logger_instance.save_cumulative_metrics()

    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a multi-agent D3QN system for traffic light control.')
    parser.add_argument('--config', type=str, default='default',
                       help='Name of the reward configuration file to use (without .json extension).')
    parser.add_argument('--checkpoint_path', type=str,
                       help='Path to a model checkpoint to resume training.')
    parser.add_argument('--start_episode', type=int, default=1, help='Episode to start training from.')
    parser.add_argument('--network_topology', type=str, default='4x4-Lucas',
                       help='Network topology to use (directory name in sumo_rl/nets)')
    parser.add_argument('--validate_freq', type=int, default=None,
                       help='Validate agent every N episodes (set to 0 to disable)')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed for reproducible training')
    parser.add_argument('--episodes', type=int,
                       help='Override the number of episodes from config')
    args = parser.parse_args()

    # Update episodes in config if provided
    if args.episodes:
        TRAINING_CONFIG['episodes'] = args.episodes

    config_path = os.path.join('configs', f'{args.config}.json')
    if not os.path.exists(config_path):
        sys.exit(f"Error: Configuration file not found at {config_path}")

    with open(config_path, 'r') as f:
        REWARD_CONFIG = json.load(f)

    logger.info(f"--- Using reward configuration: {args.config} ---")
    logger.info(f"--- Network topology: {args.network_topology} ---")

    train(
        REWARD_CONFIG,
        checkpoint_path=args.checkpoint_path,
        start_episode=args.start_episode,
        network_topology=args.network_topology,
        validate_freq=args.validate_freq,
        seed=args.seed
    )
