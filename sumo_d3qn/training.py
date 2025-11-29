
import os
import sys
import numpy as np
import torch
import sumo_rl
from datetime import datetime
import json
import argparse
import signal
from pathlib import Path
import importlib.resources as pkg_resources
import pandas as pd

from agent import D3QNAgent
from config import SUMO_CONFIG, AGENT_CONFIG, TRAINING_CONFIG, SCHEDULER_CONFIG

def save_checkpoint(episode, agents, log_dir):
    """Saves a training checkpoint for all agents."""
    checkpoint_data = {'episode': episode, 'epsilon': agents[list(agents.keys())[0]].epsilon}
    for ts_id, agent in agents.items():
        checkpoint_data[f'model_state_dict_{ts_id}'] = agent.qnetwork_local.state_dict()
        checkpoint_data[f'optimizer_state_dict_{ts_id}'] = agent.optimizer.state_dict()
    
    checkpoint_path = os.path.join(log_dir, f"multi_agent_checkpoint_ep{episode}.pth")
    torch.save(checkpoint_data, checkpoint_path)
    print(f"\nCheckpoint saved to {checkpoint_path}")

def train_d3qn_agents(checkpoint_path=None, start_episode=1, force_epsilon=None):
    """Trains D3QN agents on a multi-intersection environment."""
    log_dir = os.path.join(TRAINING_CONFIG["log_dir"], f"multi_agent_training_{datetime.now().strftime('%Y%m%d-%H%M%S')}")
    os.makedirs(log_dir, exist_ok=True)
    
    env = sumo_rl.SumoEnvironment(
        net_file=SUMO_CONFIG["net_file"],
        route_file=SUMO_CONFIG["route_file"],
        # out_csv_name=os.path.join(log_dir, f"output_episode_{episode}.csv"),
        num_seconds=SUMO_CONFIG["num_seconds"],
        delta_time=SUMO_CONFIG["delta_time"],
        yellow_time=SUMO_CONFIG["yellow_time"],
        min_green=SUMO_CONFIG["min_green"],
        single_agent=False, # Multi-agent setting
        fixed_ts=False, # Allow agents to control traffic lights
        use_gui=SUMO_CONFIG["gui"]
    )
    
    ts_ids = env.ts_ids
    agents = {}
    schedulers = {}
    for ts_id in ts_ids:
        state_size = env.observation_spaces(ts_id).shape[0]
        action_size = env.action_spaces(ts_id).n
        print(f"Initialized Agent for {ts_id}. State size: {state_size}, Action size: {action_size}")
        agent = D3QNAgent(state_size=state_size, action_size=action_size)
        agents[ts_id] = agent
        schedulers[ts_id] = torch.optim.lr_scheduler.StepLR(agent.optimizer, step_size=SCHEDULER_CONFIG["step_size"], gamma=SCHEDULER_CONFIG["gamma"])

    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path)
        for ts_id in ts_ids:
            agents[ts_id].qnetwork_local.load_state_dict(checkpoint[f'model_state_dict_{ts_id}'])
            agents[ts_id].qnetwork_target.load_state_dict(checkpoint[f'model_state_dict_{ts_id}'])
            agents[ts_id].optimizer.load_state_dict(checkpoint[f'optimizer_state_dict_{ts_id}'])
        agents[list(agents.keys())[0]].epsilon = checkpoint['epsilon'] # Epsilon is shared
        print(f"Resumed training from checkpoint: {checkpoint_path}")
        if force_epsilon is not None:
            for ts_id in ts_ids:
                agents[ts_id].epsilon = force_epsilon
            print(f"Epsilon manually overridden to: {agents[list(agents.keys())[0]].epsilon:.4f}")

    best_rewards = {ts_id: -np.inf for ts_id in ts_ids}

    for episode in range(start_episode, TRAINING_CONFIG["episodes"] + 1):
        initial_obs = env.reset()
        obs = {ts_id: initial_obs[ts_id] for ts_id in ts_ids}
        done = {"__all__": False}
        total_episode_rewards = {ts_id: 0 for ts_id in ts_ids}

        episode_metrics = [] # Initialize list to store metrics for the current episode

        while not done["__all__"]:
            action_dict = {}
            for ts_id in ts_ids:
                action_dict[ts_id] = agents[ts_id].act(obs[ts_id], agents[ts_id].epsilon)
            
            next_obs, rewards, terminated, info = env.step(action_dict)
            done["__all__"] = terminated["__all__"]
            
            # Collect metrics for the current step
            step_stats = {
                'sim_step': env.sim_step,
                'simulation_time': env.sim_step * env.delta_time,
                'system_total_stopped': info.get('system_total_stopped', 0),
                'system_total_waiting_time': info.get('system_total_waiting_time', 0),
                'system_mean_waiting_time': info.get('system_mean_waiting_time', 0),
                'system_mean_speed': info.get('system_mean_speed', 0),
                'system_mean_queue_length': info.get('system_mean_queue_length', 0)
            }
            for ts_id in ts_ids:
                step_stats[f'queue_length_{ts_id}'] = env.traffic_signals[ts_id].get_total_queued()
            episode_metrics.append(step_stats)

            for ts_id in ts_ids:
                agents[ts_id].step(obs[ts_id], action_dict[ts_id], rewards[ts_id], next_obs[ts_id], done["__all__"])
                total_episode_rewards[ts_id] += rewards[ts_id]
            
            obs = next_obs

        # Save episode metrics to CSV
        df_episode = pd.DataFrame(episode_metrics)
        episode_csv_path = os.path.join(log_dir, f"episode_{episode}_metrics.csv")
        df_episode.to_csv(episode_csv_path, index=False)
        print(f"Saved episode {episode} metrics to {episode_csv_path}")

        # Epsilon decay (shared across agents)
        for ts_id in ts_ids:
            agents[ts_id].epsilon = max(AGENT_CONFIG["epsilon_end"], AGENT_CONFIG["epsilon_decay"] * agents[ts_id].epsilon)
        
        # Call scheduler.step() for each agent after the episode
        for ts_id in ts_ids:
            schedulers[ts_id].step()

        current_lr = agents[list(agents.keys())[0]].optimizer.param_groups[0]['lr']
        print(f"Episode {episode}/{TRAINING_CONFIG['episodes']} | Avg Reward: {np.mean(list(total_episode_rewards.values())):.2f} | Epsilon: {agents[list(agents.keys())[0]].epsilon:.4f} | LR: {current_lr:.6f}")

        # Save best models for each agent
        for ts_id in ts_ids:
            if total_episode_rewards[ts_id] > best_rewards[ts_id]:
                best_rewards[ts_id] = total_episode_rewards[ts_id]
                final_model_path = os.path.join("models", f"d3qn_agent_{ts_id}.pth")
                os.makedirs(os.path.dirname(final_model_path), exist_ok=True)
                agents[ts_id].save(final_model_path)
                print(f"New best model saved for {ts_id} with reward: {best_rewards[ts_id]:.2f}")

        if episode % TRAINING_CONFIG["save_checkpoint_every"] == 0:
            save_checkpoint(episode, agents, log_dir)

    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train D3QN agents on a multi-intersection environment.')
    parser.add_argument('--checkpoint_path', type=str, help='Path to a model checkpoint to resume training.')
    parser.add_argument('--start_episode', type=int, default=1, help='Episode to start training from.')
    parser.add_argument('--episodes', type=int, default=TRAINING_CONFIG["episodes"], help='Set the total number of episodes to train for.')
    parser.add_argument('--force_epsilon', type=float, help='Manually override the epsilon value loaded from a checkpoint.')
    args = parser.parse_args()

    TRAINING_CONFIG['episodes'] = args.episodes # Update TRAINING_CONFIG with parsed episodes
    
    train_d3qn_agents(checkpoint_path=args.checkpoint_path, start_episode=args.start_episode, force_epsilon=args.force_epsilon)
