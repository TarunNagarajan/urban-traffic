import pandas as pd
import matplotlib.pyplot as plt
import os
import xml.etree.ElementTree as ET

def compute_and_print_statistics(details_df, sim_df, label):
    """
    Computes and prints summary statistics from the evaluation data.
    """
    avg_reward = details_df['reward'].mean()
    total_waiting_time = sim_df['system_total_waiting_time'].iloc[-1]
    avg_queue_length = sim_df['system_mean_queue_length'].mean()
    
    print(f"\n--- {label} Statistics ---")
    print(f"Average Step Reward: {avg_reward:.2f}")
    print(f"Total System Waiting Time: {total_waiting_time:.2f} seconds")
    print(f"Average System-wide Queue Length: {avg_queue_length:.2f} vehicles")

def plot_rolling_reward(rl_details_csv, baseline_details_csv, output_dir):
    """
    Plots the rolling average of rewards for both the D3QN agent and the baseline.
    """
    rl_df = pd.read_csv(rl_details_csv)
    baseline_df = pd.read_csv(baseline_details_csv)

    plt.figure(figsize=(12, 6))
    plt.plot(rl_df['step'], rl_df['reward'].rolling(window=10).mean(), label='D3QN Agent')
    plt.plot(baseline_df['step'], baseline_df['reward'].rolling(window=10).mean(), label='Fixed-Time Baseline')
    plt.xlabel('Simulation Step')
    plt.ylabel('Rolling Average Reward (10 steps)')
    plt.title('D3QN Agent vs. Fixed-Time Baseline: Rolling Reward')
    plt.legend()
    plt.grid(True)
    plot_path = os.path.join(output_dir, 'rolling_reward_comparison.png')
    plt.savefig(plot_path)
    print(f"Saved rolling reward plot to {plot_path}")
    plt.close()

def plot_average_speed(rl_sim_csv, baseline_sim_csv, output_dir):
    """
    Placeholder function to plot average system speed.
    Your teammate can implement this.
    """
    print("\nSkipping average speed plot (not implemented).")
    # Example implementation hint:
    # rl_df = pd.read_csv(rl_sim_csv, sep=';')
    # baseline_df = pd.read_csv(baseline_sim_csv, sep=';')
    # plt.plot(rl_df['step'], rl_df['system_mean_speed'], label='D3QN Agent')
    # ... (complete the plotting logic)

def plot_fuel_consumption(emissions_xml, output_dir):
    """
    Placeholder function to plot fuel consumption.
    Your teammate can implement this.
    """
    print("Skipping fuel consumption plot (not implemented).")
    # Example implementation hint:
    # tree = ET.parse(emissions_xml)
    # root = tree.getroot()
    # fuel_data = []
    # for timestep in root.findall('timestep'):
    #     time = float(timestep.get('time'))
    #     total_fuel = 0
    #     vehicle_count = 0
    #     for vehicle in timestep.findall('vehicle'):
    #         total_fuel += float(vehicle.get('fuel'))
    #         vehicle_count += 1
    #     if vehicle_count > 0:
    #         avg_fuel = total_fuel / vehicle_count
    #         fuel_data.append({'time': time, 'avg_fuel': avg_fuel})
    # ... (process and plot the data)


if __name__ == "__main__":
    # Directory where the evaluation output is stored
    eval_log_dir = "logs/evaluation"

    # --- File Paths ---
    rl_details_csv = os.path.join(eval_log_dir, "d3qn_details.csv")
    baseline_details_csv = os.path.join(eval_log_dir, "baseline_details.csv")
    rl_sim_csv = os.path.join(eval_log_dir, "d3qn_results.csv")
    baseline_sim_csv = os.path.join(eval_log_dir, "baseline_results.csv")
    emissions_xml = os.path.join(eval_log_dir, "emissions.xml")

    if not all(os.path.exists(f) for f in [rl_details_csv, baseline_details_csv, rl_sim_csv, baseline_sim_csv, emissions_xml]):
        print("Error: Not all required evaluation files were found.")
        print("Please run 'python evaluation.py' first to generate the data.")
    else:
        print("Generating advanced statistics and plots...")
        
        # --- Compute and Print Statistics ---
        rl_details_df = pd.read_csv(rl_details_csv)
        baseline_details_df = pd.read_csv(baseline_details_csv)
        rl_sim_df = pd.read_csv(rl_sim_csv, sep=';')
        baseline_sim_df = pd.read_csv(baseline_sim_csv, sep=';')
        
        compute_and_print_statistics(rl_details_df, rl_sim_df, "D3QN Agent")
        compute_and_print_statistics(baseline_details_df, baseline_sim_df, "Fixed-Time Baseline")

        # --- Plot Rolling Reward (Implemented Example) ---
        plot_rolling_reward(rl_details_csv, baseline_details_csv, eval_log_dir)

        # --- Plot Average Speed (Placeholder) ---
        plot_average_speed(rl_sim_csv, baseline_sim_csv, eval_log_dir)

        # --- Plot Fuel Consumption (Placeholder) ---
        plot_fuel_consumption(emissions_xml, eval_log_dir)

        print("\nAdvanced analysis complete.")