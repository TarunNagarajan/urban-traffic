import os
import pandas as pd
import matplotlib.pyplot as plt
import sumo_rl
import sumolib
import torch
import numpy as np
import csv
import json
import argparse

from agent import D3QNAgent
from config import SUMO_CONFIG, AGENT_CONFIG, TRAINING_CONFIG, REWARD_CONFIG, STUB_GRU_PREDICTION, TUNING_CONFIG
# Import functions from train.py
from train import compute_state, get_neighboring_traffic_lights, compute_reward, NUM_PHASES, PHASE_START, PHASE_END, QUEUE_START, QUEUE_END

def run_evaluation(env, net, REWARD_CONFIG, agent=None, max_neighbors=0, details_output_path=None):
    """
    Runs a single evaluation loop for a given environment and agent/baseline.
    Returns a dictionary with the detailed reward breakdown.
    """
    obs = env.reset()
    done = {"__all__": False}
    ts_ids = list(obs.keys())
    
    step = 0
    prev_arrived = 0
    episode_rewards = {}
    commitment_counters = {ts_id: 0 for ts_id in ts_ids}
    
    if details_output_path:
        with open(details_output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            header = ['step', 'total_reward']
            for ts_id in ts_ids:
                header.extend([f'{ts_id}_phase', f'{ts_id}_total_queue'])
            writer.writerow(header)

    prev_obs = obs
    while not done["__all__"]:
        # Determine dynamic queue threshold based on traffic density
        if TUNING_CONFIG.get("use_adaptive_thresholds", False):
            num_vehicles = env.sumo.vehicle.getIDCount()
            regimes = TUNING_CONFIG.get("density_regimes", [150, 300])
            thresholds = TUNING_CONFIG.get("queue_override_thresholds", [40, 30, 20])
            
            if num_vehicles < regimes[0]: # Low density
                dynamic_queue_threshold = thresholds[0]
            elif num_vehicles < regimes[1]: # Medium density
                dynamic_queue_threshold = thresholds[1]
            else: # High density
                dynamic_queue_threshold = thresholds[2]
        else:
            dynamic_queue_threshold = TUNING_CONFIG.get("queue_override_threshold", 50)

        if agent is not None:
            action_dict = {}
            for ts_id in ts_ids:
                current_phase = np.argmax(obs[ts_id][PHASE_START:PHASE_END])
                meta_action = 0 # Default to HOLD

                # --- Performance Tuning Layer (Prioritized) ---
                # 1. Check for Phase Commitment
                if TUNING_CONFIG.get("use_phase_commitment", False) and commitment_counters[ts_id] > 0:
                    meta_action = 0 # Force HOLD
                    commitment_counters[ts_id] -= 1
                else:
                    # If not committed, let agent decide
                    # Minimum green time constraint
                    if env.traffic_signals[ts_id].time_since_last_phase_change < env.traffic_signals[ts_id].min_green:
                        action_mask = np.array([1, 0])
                    else:
                        action_mask = np.array([1, 1])

                    # Get agent's suggested action
                    temperature = 0.0
                    if TUNING_CONFIG.get("use_temperature_sampling", False):
                        temperature = TUNING_CONFIG.get("inference_temperature", 0.0)
                    
                    state = compute_state(net, ts_id, obs, STUB_GRU_PREDICTION["inflow_dimension"], max_neighbors, env)
                    meta_action = agent.act(state, eps=0.0, action_mask=action_mask, temperature=temperature)
                    
                    # 2. Apply Switching Cooldown
                    if TUNING_CONFIG.get("use_switching_cooldown", False) and meta_action == 1:
                        if env.traffic_signals[ts_id].time_since_last_phase_change < TUNING_CONFIG.get("switching_cooldown_seconds", 0):
                            meta_action = 0  # Override to HOLD

                    # 3. Apply Queue Override
                    if TUNING_CONFIG.get("use_queue_override", False) and meta_action == 0:
                        queues = obs[ts_id][QUEUE_START:QUEUE_END]
                        if np.max(queues) > dynamic_queue_threshold:
                            if action_mask[1] == 1: # Check if switching is allowed
                               meta_action = 1 # Override to SWITCH

                # Set commitment counter if a switch is happening
                if meta_action == 1:
                    commitment_counters[ts_id] = TUNING_CONFIG.get("phase_commitment_steps", 1)

                # Translate meta-action to environment action
                if meta_action == 0:
                    action_dict[ts_id] = current_phase
                else:
                    action_dict[ts_id] = (current_phase + 1) % NUM_PHASES
        else:
            action_dict = {}

        next_obs, _, done, _ = env.step(action_dict)
        
        reward_info = compute_reward(env, next_obs, prev_obs, REWARD_CONFIG, prev_arrived)
        prev_arrived = env.sumo.simulation.getArrivedNumber()

        for key, value in reward_info.items():
            episode_rewards[key] = episode_rewards.get(key, 0) + value
        
        if details_output_path:
            with open(details_output_path, 'a', newline='') as f:
                writer = csv.writer(f)
                row = [step, reward_info['total_reward']]
                for ts_id in ts_ids:
                    if ts_id in obs:
                        phase = np.argmax(obs[ts_id][PHASE_START:PHASE_END])
                        total_queue = np.sum(obs[ts_id][QUEUE_START:QUEUE_END])
                        row.extend([phase, total_queue])
                    else:
                        row.extend([-1, -1])
                writer.writerow(row)

        prev_obs = obs
        obs = next_obs
        step += 1
                
    return env.out_csv_name, episode_rewards

def plot_results(rl_csv, baseline_csv, log_dir):
    """
    Loads simulation results from CSVs and plots a comparison.
    """
    rl_df = pd.read_csv(rl_csv, sep=';')
    baseline_df = pd.read_csv(baseline_csv, sep=';')

    plt.figure(figsize=(12, 6))
    plt.plot(rl_df['step'], rl_df['system_total_waiting_time'], label='D3QN Agent')
    plt.plot(baseline_df['step'], baseline_df['system_total_waiting_time'], label='Fixed-Time Baseline')
    plt.xlabel('Simulation Step')
    plt.ylabel('Total System Waiting Time (s)')
    plt.title('D3QN Agent vs. Fixed-Time Baseline: Waiting Time')
    plt.legend()
    plt.grid(True)
    plot_path = os.path.join(log_dir, 'waiting_time_comparison.png')
    plt.savefig(plot_path)
    print(f"Saved waiting time plot to {plot_path}")
    plt.close()

    plt.figure(figsize=(12, 6))
    plt.plot(rl_df['step'], rl_df['system_mean_queue_length'], label='D3QN Agent')
    plt.plot(baseline_df['step'], baseline_df['system_mean_queue_length'], label='Fixed-Time Baseline')
    plt.xlabel('Simulation Step')
    plt.ylabel('Average Queue Length (vehicles)')
    plt.title('D3QN Agent vs. Fixed-Time Baseline: Queue Length')
    plt.legend()
    plt.grid(True)
    plot_path = os.path.join(log_dir, 'queue_length_comparison.png')
    plt.savefig(plot_path)
    print(f"Saved queue length plot to {plot_path}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate a D3QN agent for traffic light control.')
    parser.add_argument('--config', type=str, default='default', help='Name of the reward configuration file to use (without .json extension).')
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to a model checkpoint for evaluation.')
    args = parser.parse_args()

    config_path = os.path.join('configs', f'{args.config}.json')
    if not os.path.exists(config_path):
        sys.exit(f"Error: Configuration file not found at {config_path}")

    with open(config_path, 'r') as f:
        REWARD_CONFIG = json.load(f)

    print(f"--- Using reward configuration: {args.config} ---")

    log_dir = TRAINING_CONFIG["log_dir"]
    eval_log_dir = os.path.join(log_dir, "evaluation")
    os.makedirs(eval_log_dir, exist_ok=True)

    env = sumo_rl.SumoEnvironment(
        net_file='C:/Users/ultim/anaconda3/envs/metaworld-cpu/lib/site-packages/sumo_rl/nets/4x4-Lucas/4x4.net.xml',
        route_file='data/4x4_diverse.rou.xml',
        out_csv_name=os.path.join(eval_log_dir, "temp_output.csv"),
        use_gui=False,
        num_seconds=SUMO_CONFIG["num_seconds"],
        delta_time=SUMO_CONFIG["delta_time"],
        yellow_time=SUMO_CONFIG["yellow_time"],
        min_green=SUMO_CONFIG.get("min_green", 10),
        max_depart_delay=0,
        additional_sumo_cmd=f"--emission-output {os.path.join(eval_log_dir, 'emissions.xml')}"
    )
    
    net = sumolib.net.readNet(env._net)

    # --- D3QN Agent Evaluation ---
    print("Evaluating D3QN Agent...")
    
    initial_obs = env.reset()
    ts_ids = list(initial_obs.keys())
    
    max_neighbors = 0
    for ts_id in ts_ids:
        neighbors = get_neighboring_traffic_lights(net, ts_id)
        if len(neighbors) > max_neighbors:
            max_neighbors = len(neighbors)

    first_ts_id = ts_ids[0]
    state_size = compute_state(net, first_ts_id, initial_obs, STUB_GRU_PREDICTION["inflow_dimension"], max_neighbors, env).shape[0]
    
    agent = D3QNAgent(state_size=state_size, action_size=AGENT_CONFIG["action_size"])
    model_path = args.checkpoint_path
    
    rl_csv_path = None
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path)
        agent.qnetwork_local.load_state_dict(checkpoint['model_state_dict'])
        env.out_csv_name = os.path.join(eval_log_dir, "d3qn_results.csv")
        rl_details_path = os.path.join(eval_log_dir, "d3qn_details.csv")
        rl_csv_path, episode_rewards = run_evaluation(env, net, REWARD_CONFIG, agent, max_neighbors, rl_details_path)
        print(f"D3QN Agent evaluation complete. Results saved to {rl_csv_path}")
        print("\n--- Agent Performance Breakdown ---")
        reward_summary = " | ".join([f"{key}: {value:.2f}" for key, value in episode_rewards.items()])
        print(reward_summary)

    else:
        print(f"Error: Model not found at {model_path}.")

    # --- Fixed-Time Baseline Evaluation ---
    print("\nEvaluating Fixed-Time Baseline...")
    env.out_csv_name = os.path.join(eval_log_dir, "baseline_results.csv")
    baseline_details_path = os.path.join(eval_log_dir, "baseline_details.csv")
    baseline_csv_path, _ = run_evaluation(env, net, REWARD_CONFIG, agent=None, details_output_path=baseline_details_path)
    print(f"Baseline evaluation complete. Results saved to {baseline_csv_path}")

    # --- Plotting ---
    if rl_csv_path and baseline_csv_path:
        print("\nGenerating comparison plots...")
        plot_results(rl_csv_path, baseline_csv_path, eval_log_dir)

    env.close()