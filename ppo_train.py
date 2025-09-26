
import os
import sys
import numpy as np
import torch
import sumo_rl
import sumolib
from datetime import datetime
import json
import argparse

from ppo_agent import PPOAgent, ActorCritic
from config import SUMO_CONFIG, AGENT_CONFIG, REWARD_CONFIG, STUB_GRU_PREDICTION
# Import functions from train.py
from train import compute_state, get_neighboring_traffic_lights, compute_reward, NUM_PHASES, PHASE_START, PHASE_END, QUEUE_START, QUEUE_END

# --- PPO Hyperparameters ---
PPO_TRAIN_CONFIG = {
    "total_timesteps": 1e6,
    "num_steps_per_update": 2048, # Steps per policy update
    "lr": 3e-4,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "ppo_clip": 0.2,
    "num_epochs": 10,
    "batch_size": 64,
    "log_dir": "logs/ppo/",
    "model_save_path": "models/ppo_agent.pth",
}

def compute_gae(next_value, rewards, dones, values, gamma, gae_lambda):
    """Computes Generalized Advantage Estimation."""
    advantages = torch.zeros_like(rewards)
    last_advantage = 0
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_non_terminal = 1.0 - dones[t]
            next_values = next_value
        else:
            next_non_terminal = 1.0 - dones[t+1]
            next_values = values[t+1]
        
        delta = rewards[t] + gamma * next_values * next_non_terminal - values[t]
        advantages[t] = last_advantage = delta + gamma * gae_lambda * next_non_terminal * last_advantage
    
    returns = advantages + values
    return returns, advantages

def train_ppo():
    """Main PPO training loop."""
    log_dir = os.path.join(PPO_TRAIN_CONFIG["log_dir"], datetime.now().strftime('%Y%m%d-%H%M%S'))
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(os.path.dirname(PPO_TRAIN_CONFIG["model_save_path"]), exist_ok=True)

    env = sumo_rl.SumoEnvironment(
        net_file=SUMO_CONFIG["net_file"],
        route_file=SUMO_CONFIG["route_file"],
        out_csv_name=os.path.join(log_dir, "output.csv"),
        use_gui=SUMO_CONFIG["gui"],
        num_seconds=1e6, # Long run for continuous training
        delta_time=SUMO_CONFIG["delta_time"],
        min_green=SUMO_CONFIG.get("min_green", 10),
    )
    
    net = sumolib.net.readNet(env._net)
    obs = env.reset()
    ts_ids = list(obs.keys())
    
    max_neighbors = 0
    for ts_id in ts_ids:
        neighbors = get_neighboring_traffic_lights(net, ts_id)
        if len(neighbors) > max_neighbors:
            max_neighbors = len(neighbors)

    # This assumes all agents have the same state/action size
    first_ts_id = ts_ids[0]
    state_size = compute_state(net, first_ts_id, obs, STUB_GRU_PREDICTION["inflow_dimension"], max_neighbors, env).shape[0]
    action_size = AGENT_CONFIG["action_size"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create a single agent for all traffic lights (parameter sharing)
    agent = PPOAgent(state_size, action_size, PPO_TRAIN_CONFIG["lr"], PPO_TRAIN_CONFIG["gamma"], PPO_TRAIN_CONFIG["ppo_clip"], PPO_TRAIN_CONFIG["num_epochs"], PPO_TRAIN_CONFIG["batch_size"], device)

    # --- Training Loop ---
    num_updates = int(PPO_TRAIN_CONFIG["total_timesteps"] // PPO_TRAIN_CONFIG["num_steps_per_update"])
    
    # Global state for the loop
    global_step = 0
    obs = env.reset()
    prev_arrived = 0
    episode_reward = 0
    episode = 0

    for update in range(1, num_updates + 1):
        # --- Storage for the rollout ---
        memory = {
            'states': [], 'actions': [], 'log_probs': [],
            'rewards': [], 'dones': [], 'values': []
        }

        # --- Collect Rollout ---
        for step in range(PPO_TRAIN_CONFIG["num_steps_per_update"]):
            global_step += 1
            
            action_dict = {}
            log_probs_step = []
            values_step = []
            actions_step = []

            # All agents act based on their observation
            for ts_id in ts_ids:
                state = compute_state(net, ts_id, obs, STUB_GRU_PREDICTION["inflow_dimension"], max_neighbors, env)
                state_tensor = torch.from_numpy(state).float().to(device).unsqueeze(0)
                
                with torch.no_grad():
                    action, log_prob, _, value = agent.get_action_and_value(state_tensor)
                
                memory['states'].append(state_tensor)
                memory['values'].append(value)
                memory['actions'].append(action)
                memory['log_probs'].append(log_prob)

                action_dict[ts_id] = action.item()

            next_obs, _, done, _ = env.step(action_dict)
            
            # For now, use a simplified global reward
            reward = -env.get_total_queued() / 100.0
            episode_reward += reward

            memory['rewards'].append(torch.tensor([reward]).to(device))
            memory['dones'].append(torch.tensor([done["__all__"]]).to(device))

            obs = next_obs

            if done["__all__"]:
                print(f"Global Step: {global_step} | Episode: {episode+1} | Episode Reward: {episode_reward:.2f}")
                obs = env.reset()
                episode_reward = 0
                episode += 1

        # --- Bootstrap value and compute GAE ---
        with torch.no_grad():
            # Get value of the last state
            last_state = compute_state(net, ts_ids[0], obs, STUB_GRU_PREDICTION["inflow_dimension"], max_neighbors, env)
            _, next_value = agent.policy(torch.from_numpy(last_state).float().to(device).unsqueeze(0))
        
        returns, advantages = compute_gae(next_value, memory['rewards'], memory['dones'], memory['values'], PPO_TRAIN_CONFIG["gamma"], PPO_TRAIN_CONFIG["gae_lambda"])

        # --- Update Policy ---
        states_tensor = torch.cat(memory['states'])
        actions_tensor = torch.cat(memory['actions'])
        log_probs_tensor = torch.cat(memory['log_probs'])

        agent.update(states_tensor, actions_tensor, log_probs_tensor, returns, advantages)

        # --- Logging and Saving ---
        print(f"Update #{update}/{num_updates} complete.")
        if update % 10 == 0:
            agent.save(PPO_TRAIN_CONFIG["model_save_path"])
            print(f"Model saved to {PPO_TRAIN_CONFIG['model_save_path']}")

    env.close()

if __name__ == "__main__":
    train_ppo()