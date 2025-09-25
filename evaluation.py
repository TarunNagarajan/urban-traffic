import os
import pandas as pd
import matplotlib.pyplot as plt
import sumo_rl
import sumolib
import torch
import numpy as np

from agent import D3QNAgent
from config import SUMO_CONFIG, AGENT_CONFIG, TRAINING_CONFIG, STUB_GRU_PREDICTION
# Import the state computation function and constants from train.py
from train import compute_state, get_neighboring_traffic_lights, NUM_PHASES, PHASE_START, PHASE_END

def run_evaluation(env, net, agent=None, max_neighbors=0):
    """
    Runs a single evaluation loop for a given environment and agent/baseline.

    Args:
        env (sumo_rl.SumoEnvironment): The SUMO environment.
        net (sumolib.net.Net): The SUMO network.
        agent (D3QNAgent, optional): The trained agent. If None, runs baseline.
        max_neighbors (int): The maximum number of neighbors for padding the state.
    """
    obs = env.reset()
    done = {"__all__": False}
    ts_ids = list(obs.keys())

    while not done["__all__"]:
        if agent is not None:
            action_dict = {}
            for ts_id in ts_ids:
                state = compute_state(net, ts_id, obs, STUB_GRU_PREDICTION["inflow_dimension"], max_neighbors)
                meta_action = agent.act(state, eps=0.0) # Act greedily
                
                current_phase = np.argmax(obs[ts_id][PHASE_START:PHASE_END])
                if meta_action == 0: # Hold
                    action = current_phase
                else: # Switch
                    action = (current_phase + 1) % NUM_PHASES
                action_dict[ts_id] = action
        else:
            # For baseline, step with an empty action dict
            action_dict = {}

        obs, _, done, _ = env.step(action_dict)
        
    return env.out_csv_name

def plot_results(rl_csv, baseline_csv, log_dir):
    """
    Loads simulation results from CSVs and plots a comparison.
    """
    rl_df = pd.read_csv(rl_csv, sep=';')
    baseline_df = pd.read_csv(baseline_csv, sep=';')

    # Plotting total waiting time
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

    # Plotting average queue length
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
    log_dir = TRAINING_CONFIG["log_dir"]
    eval_log_dir = os.path.join(log_dir, "evaluation")
    os.makedirs(eval_log_dir, exist_ok=True)

    env = sumo_rl.SumoEnvironment(
        net_file='C:/Users/ultim/anaconda3/envs/metaworld-cpu/lib/site-packages/sumo_rl/nets/4x4-Lucas/4x4.net.xml',
        route_file='C:/Users/ultim/anaconda3/envs/metaworld-cpu/lib/site-packages/sumo_rl/nets/4x4-Lucas/4x4c1.rou.xml',
        out_csv_name=os.path.join(eval_log_dir, "temp_output.csv"),
        use_gui=False,
        num_seconds=SUMO_CONFIG["num_seconds"],
        delta_time=SUMO_CONFIG["delta_time"],
        yellow_time=SUMO_CONFIG["yellow_time"],
        max_depart_delay=0
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
    state_size = compute_state(net, first_ts_id, initial_obs, STUB_GRU_PREDICTION["inflow_dimension"], max_neighbors).shape[0]
    
    agent = D3QNAgent(state_size=state_size, action_size=AGENT_CONFIG["action_size"])
    model_path = TRAINING_CONFIG["model_save_path"]
    
    rl_csv_path = None
    if os.path.exists(model_path):
        agent.qnetwork_local.load_state_dict(torch.load(model_path))
        env.out_csv_name = os.path.join(eval_log_dir, "d3qn_results.csv")
        rl_csv_path = run_evaluation(env, net, agent, max_neighbors)
        print(f"D3QN Agent evaluation complete. Results saved to {rl_csv_path}")
    else:
        print(f"Error: Model not found at {model_path}. Please run train.py first.")

    # --- Fixed-Time Baseline Evaluation ---
    print("Evaluating Fixed-Time Baseline...")
    env.out_csv_name = os.path.join(eval_log_dir, "baseline_results.csv")
    baseline_csv_path = run_evaluation(env, net, agent=None)
    print(f"Baseline evaluation complete. Results saved to {baseline_csv_path}")

    # --- Plotting ---
    if rl_csv_path and baseline_csv_path:
        print("\nGenerating comparison plots...")
        plot_results(rl_csv_path, baseline_csv_path, eval_log_dir)

    env.close()