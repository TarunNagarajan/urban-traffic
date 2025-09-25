import os
import sys
import numpy as np
import torch
import sumo_rl
import optuna
import gc
import json
import argparse

from agent import D3QNAgent
from config import SUMO_CONFIG, AGENT_CONFIG, TRAINING_CONFIG
from train import compute_state, get_neighboring_traffic_lights, compute_reward, NUM_PHASES, PHASE_START, PHASE_END, QUEUE_START, QUEUE_END

# Ensure SUMO_HOME is set
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

def objective(trial, REWARD_CONFIG):
    """
    The objective function for Optuna optimization.
    """
    # --- 1. Suggest Hyperparameters ---
    # Temporarily override the global AGENT_CONFIG for this trial
    
    # Category 1: Agent's Priorities (Reward Weights)
    # Only suggest weights for components that are enabled in the REWARD_CONFIG
    if REWARD_CONFIG.get("use_queue_length", False):
        REWARD_CONFIG['queue_length_weight'] = trial.suggest_float("queue_length_weight", -1.0, -0.01)
    if REWARD_CONFIG.get("use_jerk_penalty", False):
        REWARD_CONFIG['jerk_penalty'] = trial.suggest_float("jerk_penalty", -1.0, -0.01)
    if REWARD_CONFIG.get("use_virtual_pedestrian_penalty", False):
        REWARD_CONFIG['virtual_pedestrian_penalty'] = trial.suggest_float("virtual_pedestrian_penalty", -1.0, -0.01)
    if REWARD_CONFIG.get("use_fuel_consumption_penalty", False):
        REWARD_CONFIG['fuel_consumption_penalty'] = trial.suggest_float("fuel_consumption_penalty", -1.0, -0.01)
    if REWARD_CONFIG.get("use_pressure", False):
        REWARD_CONFIG['pressure_weight'] = trial.suggest_float("pressure_weight", -1.0, -0.01)
    if REWARD_CONFIG.get("use_throughput", False):
        REWARD_CONFIG['throughput_weight'] = trial.suggest_float("throughput_weight", 0.01, 10.0)

    # Category 2: Agent's Learning Style
    AGENT_CONFIG['lr'] = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    AGENT_CONFIG['gamma'] = trial.suggest_float("gamma", 0.95, 0.999)

    # Category 3: Agent's Cognitive Ability (Network Size)
    fc_units = trial.suggest_categorical("fc_units", [64, 128, 256])
    AGENT_CONFIG['fc1_units'] = fc_units
    AGENT_CONFIG['fc2_units'] = fc_units

    # --- 2. Run Training ---
    env = sumo_rl.SumoEnvironment(
        net_file='C:/Users/ultim/anaconda3/envs/metaworld-cpu/lib/site-packages/sumo_rl/nets/4x4-Lucas/4x4.net.xml',
        route_file='data/4x4_diverse.rou.xml',
        use_gui=False,
        num_seconds=900, # Shorter simulation for 12-hour run
        delta_time=SUMO_CONFIG["delta_time"],
        yellow_time=SUMO_CONFIG["yellow_time"],
        min_green=SUMO_CONFIG.get("min_green", 10),
        max_depart_delay=0
    )

    initial_obs = env.reset()
    ts_ids = list(initial_obs.keys())
    
    max_neighbors = 0
    for ts_id in ts_ids:
        neighbors = get_neighboring_traffic_lights(net, ts_id)
        if len(neighbors) > max_neighbors:
            max_neighbors = len(neighbors)

    first_ts_id = ts_ids[0]
    state_size = compute_state(net, first_ts_id, initial_obs, 0, max_neighbors, env).shape[0]
    
    # Agent will now be created with the trial-specific values from the global AGENT_CONFIG
    agent = D3QNAgent(state_size=state_size, action_size=AGENT_CONFIG["action_size"])

    total_rewards = []
    num_tune_episodes = 30

    for episode in range(1, num_tune_episodes + 1):
        obs = env.reset()
        total_reward = 0
        done = {"__all__": False}
        prev_obs = obs
        prev_arrived = 0

        while not done["__all__"]:
            action_dict = {}
            states = {}
            meta_actions = {}

            for ts_id in ts_ids:
                if env.traffic_signals[ts_id].time_since_last_phase_change < env.traffic_signals[ts_id].min_green:
                    action_mask = np.array([1, 0])
                else:
                    action_mask = np.array([1, 1])

                states[ts_id] = compute_state(net, ts_id, obs, 0, max_neighbors, env)
                meta_actions[ts_id] = agent.act(states[ts_id], agent.epsilon, action_mask)
                
                current_phase = np.argmax(obs[ts_id][PHASE_START:PHASE_END])
                if meta_actions[ts_id] == 0:
                    action_dict[ts_id] = current_phase
                else:
                    action_dict[ts_id] = (current_phase + 1) % NUM_PHASES

            next_obs, _, done, _ = env.step(action_dict)

            reward = compute_reward(env, next_obs, prev_obs, REWARD_CONFIG, prev_arrived)
            prev_arrived = env.sumo.simulation.getArrivedNumber()

            for ts_id in ts_ids:
                next_state = compute_state(net, ts_id, next_obs, 0, max_neighbors, env)
                agent.step(states[ts_id], meta_actions[ts_id], reward, next_state, done["__all__"])

            prev_obs = obs
            obs = next_obs
            total_reward += reward

        agent.epsilon = max(AGENT_CONFIG["epsilon_end"], AGENT_CONFIG["epsilon_decay"] * agent.epsilon)
        total_rewards.append(total_reward)

    env.close()
    score = np.mean(total_rewards[-5:])

    del env, agent
    gc.collect()

    return score

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Tune D3QN agent hyperparameters for traffic light control.')
    parser.add_argument('--config', type=str, default='default', help='Name of the reward configuration file to use (without .json extension).')
    args = parser.parse_args()

    config_path = os.path.join('configs', f'{args.config}.json')
    if not os.path.exists(config_path):
        sys.exit(f"Error: Configuration file not found at {config_path}")

    with open(config_path, 'r') as f:
        BASE_REWARD_CONFIG = json.load(f)
    
    print(f"--- Using base reward configuration: {args.config} for tuning ---")

    results_dir = "sumo_d3qn/tuning_results"
    os.makedirs(results_dir, exist_ok=True)

    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, BASE_REWARD_CONFIG), n_trials=40) # Set for ~12-hour run

    print("\n--- Optuna Study Complete ---")
    print(f"Number of finished trials: {len(study.trials)}")
    print("Best trial:")
    trial = study.best_trial

    print(f"  Value (Score): {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    best_params_path = os.path.join(results_dir, f"best_hyperparameters_{args.config}.txt")
    with open(best_params_path, "w") as f:
        f.write("Best Hyperparameters Found by Optuna:\n")
        for key, value in trial.params.items():
            f.write(f"{key}: {value}\n")
        f.write(f"\nBest Score (Value): {trial.value}\n")
    print(f"Best hyperparameters saved to '{best_params_path}'")

    try:
        history_fig = optuna.visualization.plot_optimization_history(study)
        history_fig.write_html(os.path.join(results_dir, f"optimization_history_{args.config}.html"))

        importance_fig = optuna.visualization.plot_param_importances(study)
        importance_fig.write_html(os.path.join(results_dir, f"param_importances_{args.config}.html"))

        print(f"\nSuccessfully saved study plots to '{results_dir}'")
        print("You can open the .html files in your browser to view the interactive plots.")

    except (ImportError, ModuleNotFoundError):
        print(f"\nCould not generate plots. Please install plotly: pip install plotly")
    except Exception as e:
        print(f"\nAn error occurred during plot generation: {e}")
