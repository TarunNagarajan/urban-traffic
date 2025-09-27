

import os
import pandas as pd
import matplotlib.pyplot as plt
import sumo_rl
import torch
import numpy as np
import argparse

from agent import D3QNAgent
from config import SUMO_CONFIG

def run_evaluation(env, agent=None):
    """Runs a multi-agent evaluation loop and returns a pandas DataFrame of the results."""
    obs, _ = env.reset()
    done = {"__all__": False}
    ts_ids = env.ts_ids
    simulation_data = []

    while not done["__all__"]:
        action_dict = {}
        if agent is not None:
            for ts_id in ts_ids:
                state = obs[ts_id]
                action = agent.act(state, eps=0.0) # Use greedy policy for evaluation
                action_dict[ts_id] = action
        else:
            # When agent is None, sumo-rl uses its default fixed-time controller
            pass

        next_obs, _, terminated, truncated, info = env.step(action_dict)
        done = terminated

        # Manually collect the summary statistics from the `info` dictionary
        step_stats = {
            'step': env.sim_step,
            'system_total_stopped': info.get('system_total_stopped', 0),
            'system_total_waiting_time': info.get('system_total_waiting_time', 0),
            'system_mean_waiting_time': info.get('system_mean_waiting_time', 0),
            'system_mean_speed': info.get('system_mean_speed', 0),
            'system_mean_queue_length': info.get('system_mean_queue_length', 0)
        }
        simulation_data.append(step_stats)
        obs = next_obs
    
    # Manually create and save the DataFrame
    df = pd.DataFrame(simulation_data)
    df.to_csv(env.out_csv_name, sep=';', index=False)
    print(f"Saved results to {env.out_csv_name}")
    return df

def plot_results(rl_df, baseline_df, log_dir):
    """Generates and saves comparison plots."""
    plt.figure(figsize=(12, 6))
    plt.plot(rl_df['step'], rl_df['system_total_waiting_time'], label='D3QN Agent')
    plt.plot(baseline_df['step'], baseline_df['system_total_waiting_time'], label='Fixed-Time Baseline')
    plt.xlabel('Simulation Step')
    plt.ylabel('Total System Waiting Time (s)')
    plt.title('D3QN Agent vs. Baseline: Waiting Time (4x4 Grid)')
    plt.legend()
    plt.grid(True)
    plot_path = os.path.join(log_dir, '4x4_wait_time_comparison.png')
    plt.savefig(plot_path)
    print(f"Saved waiting time plot to {plot_path}")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate a D3QN agent on the 4x4 grid.')
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to the D3QN model checkpoint.')
    args = parser.parse_args()

    log_dir = "logs/"
    eval_log_dir = os.path.abspath(os.path.join(log_dir, "final_4x4_evaluation"))
    os.makedirs(eval_log_dir, exist_ok=True)

    rl_csv_path = os.path.join(eval_log_dir, "d3qn_agent_results.csv")
    baseline_csv_path = os.path.join(eval_log_dir, "baseline_results.csv")

    # --- D3QN Agent Evaluation ---
    print("--- Evaluating D3QN Agent on 4x4 Grid ---")
    agent_df = None
    if os.path.exists(args.checkpoint_path):
        env = sumo_rl.SumoEnvironment(
            net_file=SUMO_CONFIG["net_file"], route_file=SUMO_CONFIG["route_file"], out_csv_name=rl_csv_path,
            use_gui=SUMO_CONFIG["gui"], num_seconds=SUMO_CONFIG["num_seconds"], delta_time=SUMO_CONFIG["delta_time"],
            yellow_time=SUMO_CONFIG["yellow_time"], min_green=SUMO_CONFIG["min_green"]
        )
        agent = D3QNAgent(state_size=env.observation_space.spaces[env.ts_ids[0]].shape[0], action_size=env.action_space.spaces[env.ts_ids[0]].n)
        checkpoint = torch.load(args.checkpoint_path)
        if 'model_state_dict' in checkpoint:
            agent.qnetwork_local.load_state_dict(checkpoint['model_state_dict'])
        else:
            agent.qnetwork_local.load_state_dict(checkpoint)
        agent_df = run_evaluation(env, agent)
        env.close()

    # --- Fixed-Time Baseline Evaluation ---
    print("\n--- Evaluating Fixed-Time Baseline ---")
    env = sumo_rl.SumoEnvironment(
        net_file=SUMO_CONFIG["net_file"], route_file=SUMO_CONFIG["route_file"], out_csv_name=baseline_csv_path,
        use_gui=SUMO_CONFIG["gui"], num_seconds=SUMO_CONFIG["num_seconds"], delta_time=SUMO_CONFIG["delta_time"],
        yellow_time=SUMO_CONFIG["yellow_time"], min_green=SUMO_CONFIG["min_green"]
    )
    baseline_df = run_evaluation(env, agent=None)
    env.close()
    
    # --- Final Comparison ---
    print("\n--- Final Performance Comparison ---")
    if agent_df is not None and baseline_df is not None:
        agent_wait_time = agent_df['system_total_waiting_time'].iloc[-1]
        baseline_wait_time = baseline_df['system_total_waiting_time'].iloc[-1]
        
        print(f"D3QN Agent Total Waiting Time: {agent_wait_time:.2f}")
        print(f"Baseline Total Waiting Time:   {baseline_wait_time:.2f}")
        
        if baseline_wait_time > agent_wait_time:
            reduction = ((baseline_wait_time - agent_wait_time) / baseline_wait_time) * 100
            print(f"\nResult: Agent was {reduction:.2f}% better.")
        else:
            increase = ((agent_wait_time - baseline_wait_time) / baseline_wait_time) * 100
            print(f"\nResult: Agent was {increase:.2f}% worse.")

        # --- Plotting ---
        print("\n--- Generating Comparison Plots ---")
        plot_results(agent_df, baseline_df, eval_log_dir)
    else:
        print("\nCould not generate comparison. A simulation run failed.")
