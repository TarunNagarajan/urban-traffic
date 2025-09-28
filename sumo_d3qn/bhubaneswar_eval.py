
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sumo_rl import SumoEnvironment
from bhubaneswar_train import D3QN_Agent, CommunicatingSumoEnvironment

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

def run_evaluation(env, agents, baseline=False):
    """Runs a single evaluation episode.

    Args:
        env (SumoEnvironment): The SUMO environment.
        agents (dict): A dictionary of D3QN agents, one for each traffic light.
        baseline (bool): If True, runs the simulation with the default SUMO controller.

    Returns:
        pd.DataFrame: A DataFrame with the collected data.
    """
    if baseline:
        env.fixed_ts = True
    else:
        env.fixed_ts = False

    states = env.reset()
    done = {'__all__': False}
    data = []

    while not done['__all__']:
        if baseline:
            actions = None
        else:
            actions = {ts_id: agents[ts_id].act(states[ts_id].reshape(1, -1)) for ts_id in env.ts_ids}
        
        next_states, _, done, _ = env.step(action=actions)

        # Log data at each step
        step_data = {'step': env.sim_step}

        # System-wide data
        step_data.update(env._get_system_info())

        # Per-agent data
        step_data.update(env._get_per_agent_info())

        # Per-vehicle data for playback
        vehicles = env.sumo.vehicle.getIDList()
        for vehicle_id in vehicles:
            step_data[f"{vehicle_id}_pos"] = env.sumo.vehicle.getPosition(vehicle_id)
            step_data[f"{vehicle_id}_speed"] = env.sumo.vehicle.getSpeed(vehicle_id)
            step_data[f"{vehicle_id}_lane"] = env.sumo.vehicle.getLaneID(vehicle_id)
            step_data[f"{vehicle_id}_angle"] = env.sumo.vehicle.getAngle(vehicle_id)
            step_data[f"{vehicle_id}_type"] = env.sumo.vehicle.getTypeID(vehicle_id)

        # Traffic light data
        for ts_id in env.ts_ids:
            step_data[f"{ts_id}_state"] = env.sumo.trafficlight.getRedYellowGreenState(ts_id)

        # Lane data
        lanes = env.sumo.lane.getIDList()
        for lane_id in lanes:
            step_data[f"{lane_id}_occupancy"] = env.sumo.lane.getLastStepOccupancy(lane_id)
            step_data[f"{lane_id}_mean_speed"] = env.sumo.lane.getLastStepMeanSpeed(lane_id)

        # Junction data
        junctions = env.sumo.junction.getIDList()
        for junction_id in junctions:
            step_data[f"{junction_id}_pos"] = env.sumo.junction.getPosition(junction_id)

        data.append(step_data)

        states = next_states

    return pd.DataFrame(data)

if __name__ == '__main__':
    # Create the environment for evaluation
    eval_env = CommunicatingSumoEnvironment(net_file='C:/Users/ultim/finalconfig/Bhubaneswar.net.xml',
                                        route_file='C:/Users/ultim/finalconfig/Bhubaneswar.rou.xml',
                                        out_csv_name='outputs/bhubaneswar_eval',
                                        use_gui=False,
                                        num_seconds=3600,  # 1 hour for evaluation
                                        delta_time=1, # 1 second for smooth playback
                                        yellow_time=3,
                                        min_green=5,
                                        max_green=60)

    # Load the trained agents
    base_state_size = eval_env.observation_space.shape[0]
    ts_id_list = list(eval_env.ts_ids)
    state_size = base_state_size * (1 + len(eval_env.neighbors[ts_id_list[0]]))
    action_size = eval_env.action_space.n
    
    agents = {ts_id: D3QN_Agent(state_size, action_size) for ts_id in eval_env.ts_ids}
    # In a real scenario, you would load the trained weights here
    # for ts_id in eval_env.ts_ids:
    #     agents[ts_id].model.load_weights(f'models/{ts_id}_d3qn.h5')

    # Run evaluation with D3QN agents
    print("Running evaluation with D3QN agents...")
    d3qn_data = run_evaluation(eval_env, agents)
    d3qn_data.to_csv('outputs/bhubaneswar_d3qn_eval.csv', index=False)
    print("D3QN evaluation finished.")

    # Run evaluation with baseline controller
    print("Running evaluation with baseline controller...")
    baseline_data = run_evaluation(eval_env, agents, baseline=True)
    baseline_data.to_csv('outputs/bhubaneswar_baseline_eval.csv', index=False)
    print("Baseline evaluation finished.")

    # Generate and save plots
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.plot(d3qn_data['step'], d3qn_data['system_total_waiting_time'], label='D3QN')
    plt.plot(baseline_data['step'], baseline_data['system_total_waiting_time'], label='Baseline')
    plt.xlabel('Time (s)')
    plt.ylabel('Total Waiting Time (s)')
    plt.title('Total Waiting Time vs. Time')
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(d3qn_data['step'], d3qn_data['system_total_stopped'], label='D3QN')
    plt.plot(baseline_data['step'], baseline_data['system_total_stopped'], label='Baseline')
    plt.xlabel('Time (s)')
    plt.ylabel('Total Stopped Vehicles')
    plt.title('Total Stopped Vehicles vs. Time')
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(d3qn_data['step'], d3qn_data['system_mean_speed'], label='D3QN')
    plt.plot(baseline_data['step'], baseline_data['system_mean_speed'], label='Baseline')
    plt.xlabel('Time (s)')
    plt.ylabel('Average Speed (m/s)')
    plt.title('Average Speed vs. Time')
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.bar(['D3QN', 'Baseline'], [d3qn_data['system_total_arrived'].iloc[-1], baseline_data['system_total_arrived'].iloc[-1]])
    plt.ylabel('Total Arrived Vehicles')
    plt.title('Total Arrived Vehicles')

    plt.tight_layout()
    plt.savefig('outputs/bhubaneswar_evaluation.png')
    plt.show()

    eval_env.close()
