
import os
import sys
import numpy as np
import torch
import sumo_rl
from datetime import datetime
import json
import argparse
import signal

from agent import D3QNAgent
from config import SUMO_CONFIG, AGENT_CONFIG, TRAINING_CONFIG, SCHEDULER_CONFIG

def save_checkpoint(episode, agent, log_dir):
    """Saves a training checkpoint."""
    checkpoint_path = os.path.join(log_dir, f"worker_checkpoint_ep{episode}.pth")
    torch.save({
        'episode': episode,
        'model_state_dict': agent.qnetwork_local.state_dict(),
        'optimizer_state_dict': agent.optimizer.state_dict(),
        'epsilon': agent.epsilon,
    }, checkpoint_path)
    print(f"\nCheckpoint saved to {checkpoint_path}")

def train_worker_agent(checkpoint_path=None, start_episode=1, force_epsilon=None):
    """Trains the D3QN Worker agent on a single-intersection environment."""
    log_dir = os.path.join(TRAINING_CONFIG["log_dir"], f"worker_training_{datetime.now().strftime('%Y%m%d-%H%M%S')}")
    os.makedirs(log_dir, exist_ok=True)
    final_model_path = "models/worker_agent.pth"
    os.makedirs(os.path.dirname(final_model_path), exist_ok=True)

    env = sumo_rl.SumoEnvironment(
        net_file=SUMO_CONFIG["net_file"],
        route_file=SUMO_CONFIG["route_file"],
        out_csv_name=os.path.join(log_dir, "output.csv"),
        num_seconds=900,
        delta_time=5,
        yellow_time=2,
        min_green=10,
        single_agent=True
    )
    
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    print(f"Initialized Worker training environment. State size: {state_size}, Action size: {action_size}")

    agent = D3QNAgent(state_size=state_size, action_size=action_size)
    
    scheduler = torch.optim.lr_scheduler.StepLR(agent.optimizer, step_size=100, gamma=0.5)

    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path)
        agent.load_checkpoint(checkpoint)
        print(f"Resumed training from checkpoint: {checkpoint_path}")
        # Override epsilon if force_epsilon is provided
        if force_epsilon is not None:
            agent.epsilon = force_epsilon
            print(f"Epsilon manually overridden to: {agent.epsilon:.4f}")

    best_reward = -np.inf

    for episode in range(start_episode, TRAINING_CONFIG["episodes"] + 1):
        obs, _ = env.reset()
        done = False
        total_episode_reward = 0

        while not done:
            action = agent.act(obs, agent.epsilon)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            agent.step(obs, action, reward, next_obs, done)
            
            obs = next_obs
            total_episode_reward += reward

        agent.epsilon = max(AGENT_CONFIG["epsilon_end"], AGENT_CONFIG["epsilon_decay"] * agent.epsilon)
        scheduler.step()

        current_lr = agent.optimizer.param_groups[0]['lr']
        print(f"Episode {episode}/{TRAINING_CONFIG['episodes']} | Total Reward: {total_episode_reward:.2f} | Epsilon: {agent.epsilon:.4f} | LR: {current_lr:.6f}")

        if total_episode_reward > best_reward:
            best_reward = total_episode_reward
            agent.save(final_model_path)
            print(f"New best model saved with reward: {best_reward:.2f}")

        if episode % 10 == 0:
            save_checkpoint(episode, agent, log_dir)

    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train the D3QN Worker Agent.')
    parser.add_argument('--checkpoint_path', type=str, help='Path to a model checkpoint to resume training.')
    parser.add_argument('--start_episode', type=int, default=1, help='Episode to start training from.')
    parser.add_argument('--episodes', type=int, default=500, help='Set the total number of episodes to train for.')
    parser.add_argument('--force_epsilon', type=float, help='Manually override the epsilon value loaded from a checkpoint.')
    args = parser.parse_args()

    TRAINING_CONFIG['episodes'] = args.episodes
    
    train_worker_agent(checkpoint_path=args.checkpoint_path, start_episode=args.start_episode, force_epsilon=args.force_epsilon)
