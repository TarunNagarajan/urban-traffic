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
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET

from agent import D3QNAgent
from config import SUMO_CONFIG, AGENT_CONFIG

def parse_emission_xml(emission_file_path):
    """Parses the SUMO emission XML file and returns a DataFrame of emission data."""
    tree = ET.parse(emission_file_path)
    root = tree.getroot()

    emission_data = []
    for timestep in root.findall('timestep'):
        time = float(timestep.get('time'))
        for vehicle in timestep.findall('vehicle'):
            emission_data.append({
                'time': time,
                'vehicle_id': vehicle.get('id'),
                'CO': float(vehicle.get('CO')),
                'CO2': float(vehicle.get('CO2')),
                'HC': float(vehicle.get('HC')),
                'NOx': float(vehicle.get('NOx')),
                'PMx': float(vehicle.get('PMx')),
                'fuel': float(vehicle.get('fuel')),
                'noise': float(vehicle.get('noise')),
                'electricity': float(vehicle.get('electricity'))
            })
    return pd.DataFrame(emission_data)

def run_evaluation(env, agents=None, episode_name="evaluation", log_dir="logs/evaluation"):
    """
    Runs a multi-agent evaluation loop and returns a pandas DataFrame of the results.
    Collects detailed metrics including per-intersection queue lengths and emissions.
    """
    os.makedirs(log_dir, exist_ok=True)
    
    # Configure emission output for SUMO
    emission_file_path = os.path.join(log_dir, f"{episode_name}_emissions.xml")
    additional_sumo_cmd = f"--emission-output {emission_file_path}"

    # Re-initialize environment with emission output
    env.close() # Close previous environment if open
    env = sumo_rl.SumoEnvironment(
        net_file=SUMO_CONFIG["net_file"],
        route_file=SUMO_CONFIG["route_file"],
        out_csv_name=os.path.join(log_dir, f"{episode_name}_output.csv"),
        num_seconds=SUMO_CONFIG["num_seconds"],
        delta_time=SUMO_CONFIG["delta_time"],
        yellow_time=SUMO_CONFIG["yellow_time"],
        min_green=SUMO_CONFIG["min_green"],
        single_agent=False,
        fixed_ts= (agents is None), # Fixed-time if no agents, otherwise agents control
        use_gui=SUMO_CONFIG["gui"],
        additional_sumo_cmd=additional_sumo_cmd
    )

    obs, _ = env.reset()
    done = {"__all__": False}
    ts_ids = env.ts_ids
    simulation_data = []

    while not done["__all__"]:
        action_dict = {}
        if agents is not None:
            for ts_id in ts_ids:
                state = obs[ts_id]
                action = agents[ts_id].act(state, eps=0.0) # Greedy policy for evaluation
                action_dict[ts_id] = action
        else:
            # When agents is None, sumo-rl uses its default fixed-time controller
            pass

        next_obs, rewards, terminated, info = env.step(action_dict)
        done = terminated["__all__"]

        step_stats = {
            'sim_step': env.sim_step,
            'simulation_time': env.sim_step * env.delta_time,
            'system_total_stopped': info.get('system_total_stopped', 0),
            'system_total_waiting_time': info.get('system_total_waiting_time', 0),
            'system_mean_waiting_time': info.get('system_mean_waiting_time', 0),
            'system_mean_speed': info.get('system_mean_speed', 0),
            'system_mean_queue_length': info.get('system_mean_queue_length', 0)
        }
        
        # Add per-intersection queue lengths
        for ts_id in ts_ids:
            # sumo_rl environment provides queue length per traffic signal
            step_stats[f'queue_length_{ts_id}'] = env.traffic_signals[ts_id].get_queue()

        simulation_data.append(step_stats)
        obs = next_obs
    
    env.close()

    # Convert to DataFrame and save to CSV
    df = pd.DataFrame(simulation_data)
    df.to_csv(os.path.join(log_dir, f"{episode_name}_metrics.csv"), index=False)
    print(f"Saved detailed metrics to {os.path.join(log_dir, f'{episode_name}_metrics.csv')}")

    # Parse emission data
    emission_df = parse_emission_xml(emission_file_path)
    emission_df.to_csv(os.path.join(log_dir, f"{episode_name}_emissions.csv"), index=False)
    print(f"Saved emission data to {os.path.join(log_dir, f'{episode_name}_emissions.csv')}")

    return df, emission_df

def plot_results(rl_df, rl_emission_df, baseline_df, baseline_emission_df, log_dir, ts_ids):
    """Generates and saves comparison plots for various metrics."""
    
    # Plot 1: Total System Waiting Time
    plt.figure(figsize=(12, 6))
    plt.plot(rl_df['simulation_time'], rl_df['system_total_waiting_time'], label='D3QN Agent')
    plt.plot(baseline_df['simulation_time'], baseline_df['system_total_waiting_time'], label='Fixed-Time Baseline')
    plt.xlabel('Simulation Time (s)')
    plt.ylabel('Total System Waiting Time (s)')
    plt.title('D3QN Agent vs. Baseline: Total Waiting Time (4x4 Grid)')
    plt.legend()
    plt.grid(True)
    plot_path = os.path.join(log_dir, 'total_waiting_time_comparison.png')
    plt.savefig(plot_path)
    print(f"Saved total waiting time plot to {plot_path}")
    plt.close()

    # Plot 2: Mean System Queue Length
    plt.figure(figsize=(12, 6))
    plt.plot(rl_df['simulation_time'], rl_df['system_mean_queue_length'], label='D3QN Agent')
    plt.plot(baseline_df['simulation_time'], baseline_df['system_mean_queue_length'], label='Fixed-Time Baseline')
    plt.xlabel('Simulation Time (s)')
    plt.ylabel('Mean System Queue Length')
    plt.title('D3QN Agent vs. Baseline: Mean Queue Length (4x4 Grid)')
    plt.legend()
    plt.grid(True)
    plot_path = os.path.join(log_dir, 'mean_queue_length_comparison.png')
    plt.savefig(plot_path)
    print(f"Saved mean queue length plot to {plot_path}")
    plt.close()

    # Plot 3: Total CO2 Emissions
    if not rl_emission_df.empty and not baseline_emission_df.empty:
        rl_total_co2 = rl_emission_df.groupby('time')['CO2'].sum()
        baseline_total_co2 = baseline_emission_df.groupby('time')['CO2'].sum()

        plt.figure(figsize=(12, 6))
        plt.plot(rl_total_co2.index, rl_total_co2.values, label='D3QN Agent')
        plt.plot(baseline_total_co2.index, baseline_total_co2.values, label='Fixed-Time Baseline')
        plt.xlabel('Simulation Time (s)')
        plt.ylabel('Total CO2 Emissions (mg)')
        plt.title('D3QN Agent vs. Baseline: Total CO2 Emissions (4x4 Grid)')
        plt.legend()
        plt.grid(True)
        plot_path = os.path.join(log_dir, 'total_co2_emissions_comparison.png')
        plt.savefig(plot_path)
        print(f"Saved total CO2 emissions plot to {plot_path}")
        plt.close()
    else:
        print("Emission data not available for plotting.")

    # Plot 4: Per-intersection Queue Lengths (D3QN Agent)
    plt.figure(figsize=(15, 8))
    for ts_id in ts_ids:
        plt.plot(rl_df['simulation_time'], rl_df[f'queue_length_{ts_id}'], label=f'TS {ts_id}')
    plt.xlabel('Simulation Time (s)')
    plt.ylabel('Queue Length')
    plt.title('D3QN Agent: Per-Intersection Queue Lengths (4x4 Grid)')
    plt.legend()
    plt.grid(True)
    plot_path = os.path.join(log_dir, 'd3qn_per_intersection_queue_lengths.png')
    plt.savefig(plot_path)
    print(f"Saved D3QN per-intersection queue lengths plot to {plot_path}")
    plt.close()

    # Plot 5: Per-intersection Queue Lengths (Fixed-Time Baseline)
    plt.figure(figsize=(15, 8))
    for ts_id in ts_ids:
        plt.plot(baseline_df['simulation_time'], baseline_df[f'queue_length_{ts_id}'], label=f'TS {ts_id}')
    plt.xlabel('Simulation Time (s)')
    plt.ylabel('Queue Length')
    plt.title('Fixed-Time Baseline: Per-Intersection Queue Lengths (4x4 Grid)')
    plt.legend()
    plt.grid(True)
    plot_path = os.path.join(log_dir, 'baseline_per_intersection_queue_lengths.png')
    plt.savefig(plot_path)
    print(f"Saved Baseline per-intersection queue lengths plot to {plot_path}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate D3QN agents on the 4x4 grid.')
    parser.add_argument('--model_path', type=str, help='Path to the directory containing trained D3QN models (e.g., models/d3qn_agent_TS1.pth).')
    parser.add_argument('--gui', action='store_true', help='Enable GUI for SUMO simulation.')
    args = parser.parse_args()

    # Override GUI setting from config if specified in arguments
    SUMO_CONFIG["gui"] = args.gui

    eval_log_dir = os.path.abspath(os.path.join("logs", "final_4x4_evaluation", datetime.now().strftime('%Y%m%d-%H%M%S')))
    os.makedirs(eval_log_dir, exist_ok=True)

    # Initialize a dummy environment to get ts_ids and observation/action spaces
    # This environment will be closed and re-initialized in run_evaluation
    temp_env = sumo_rl.SumoEnvironment(
        net_file=SUMO_CONFIG["net_file"],
        route_file=SUMO_CONFIG["route_file"],
        out_csv_name=os.path.join(eval_log_dir, "temp_init.csv"),
        num_seconds=1, # Short duration for initialization
        delta_time=SUMO_CONFIG["delta_time"],
        single_agent=False,
        fixed_ts=False,
        use_gui=False
    )
    ts_ids = temp_env.ts_ids
    temp_env.close()

    # --- D3QN Agent Evaluation ---
    print("--- Evaluating D3QN Agent on 4x4 Grid ---")
    agents = {}
    if args.model_path:
        for ts_id in ts_ids:
            model_file = os.path.join(args.model_path, f"d3qn_agent_{ts_id}.pth")
            if os.path.exists(model_file):
                agent = D3QNAgent(state_size=temp_env.observation_space(ts_id).shape[0], action_size=temp_env.action_space(ts_id).n)
                agent.qnetwork_local.load_state_dict(torch.load(model_file))
                agent.qnetwork_local.eval() # Set to evaluation mode
                agents[ts_id] = agent
                print(f"Loaded model for {ts_id} from {model_file}")
            else:
                print(f"Warning: Model for {ts_id} not found at {model_file}. Skipping D3QN evaluation.")
                agents = None # If any model is missing, skip D3QN evaluation
                break
    else:
        print("No model path provided. Skipping D3QN agent evaluation.")
        agents = None

    rl_df = None
    rl_emission_df = None
    if agents:
        rl_df, rl_emission_df = run_evaluation(env=temp_env, agents=agents, episode_name="d3qn_agent", log_dir=eval_log_dir)

    # --- Fixed-Time Baseline Evaluation ---
    print("
--- Evaluating Fixed-Time Baseline ---")
    baseline_df, baseline_emission_df = run_evaluation(env=temp_env, agents=None, episode_name="fixed_time_baseline", log_dir=eval_log_dir)
    
    # --- Final Comparison and Plotting ---
    print("\n--- Final Performance Comparison and Plotting ---")
    if rl_df is not None and baseline_df is not None:
        plot_results(rl_df, rl_emission_df, baseline_df, baseline_emission_df, eval_log_dir, ts_ids)
    else:
        print("
Could not generate comparison plots. One or both simulation runs failed or were skipped.")

    print(f"
Evaluation complete. Results saved in: {eval_log_dir}")