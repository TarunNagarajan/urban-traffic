import os
import sys
import numpy as np
import torch
import sumo_rl
import sumolib
import optuna
import gc

from agent import D3QNAgent
from config import SUMO_CONFIG, AGENT_CONFIG, TRAINING_CONFIG
from train import compute_state, get_neighboring_traffic_lights, NUM_PHASES, PHASE_START, PHASE_END, QUEUE_START, QUEUE_END

# Ensure SUMO_HOME is set
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

def objective(trial):
    """
    The objective function for Optuna optimization.
    """
    # --- 1. Suggest Hyperparameters ---
    # Temporarily override the global AGENT_CONFIG for this trial
    # This ensures the D3QNAgent constructor uses the trial-specific values
    
    # Category 1: Agent's Priorities (Reward Weights)
    reward_config = {
        "queue_length_weight": trial.suggest_float("queue_length_weight", -1.0, -0.01),
        "jerk_penalty": trial.suggest_float("jerk_penalty", -1.0, -0.01),
    }

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
        route_file='C:/Users/ultim/anaconda3/envs/metaworld-cpu/lib/site-packages/sumo_rl/nets/4x4-Lucas/4x4c1.rou.xml',
        use_gui=False,
        num_seconds=900, # Shorter simulation for 12-hour run
        delta_time=SUMO_CONFIG["delta_time"],
        yellow_time=SUMO_CONFIG["yellow_time"],
        max_depart_delay=0
    )
    
    net = sumolib.net.readNet(env._net)

    initial_obs = env.reset()
    ts_ids = list(initial_obs.keys())
    
    max_neighbors = 0
    for ts_id in ts_ids:
        neighbors = get_neighboring_traffic_lights(net, ts_id)
        if len(neighbors) > max_neighbors:
            max_neighbors = len(neighbors)

    first_ts_id = ts_ids[0]
    state_size = compute_state(net, first_ts_id, initial_obs, 0, max_neighbors).shape[0]
    
    # Agent will now be created with the trial-specific values from the global AGENT_CONFIG
    agent = D3QNAgent(state_size=state_size, action_size=AGENT_CONFIG["action_size"])

    epsilon = AGENT_CONFIG["epsilon_start"]
    total_rewards = []
    num_tune_episodes = 30

    for episode in range(1, num_tune_episodes + 1):
        obs = env.reset()
        total_reward = 0
        done = {"__all__": False}

        while not done["__all__"]:
            action_dict = {}
            states = {}
            meta_actions = {}

            for ts_id in ts_ids:
                states[ts_id] = compute_state(net, ts_id, obs, 0, max_neighbors)
                meta_actions[ts_id] = agent.act(states[ts_id], epsilon)
                
                current_phase = np.argmax(obs[ts_id][PHASE_START:PHASE_END])
                if meta_actions[ts_id] == 0: # Hold
                    action_dict[ts_id] = current_phase
                else: # Switch
                    action_dict[ts_id] = (current_phase + 1) % NUM_PHASES

            next_obs, _, done, _ = env.step(action_dict)

            # Calculate reward using the trial-specific reward_config
            reward = 0
            total_queue = 0
            for ts_id in next_obs.keys():
                total_queue += np.sum(next_obs[ts_id][QUEUE_START:QUEUE_END])
            reward += total_queue * reward_config["queue_length_weight"]

            for ts_id in next_obs.keys():
                if np.argmax(next_obs[ts_id][PHASE_START:PHASE_END]) != np.argmax(obs[ts_id][PHASE_START:PHASE_END]):
                    reward += reward_config["jerk_penalty"]

            for ts_id in ts_ids:
                next_state = compute_state(net, ts_id, next_obs, 0, max_neighbors)
                agent.step(states[ts_id], meta_actions[ts_id], reward, next_state, done["__all__"])

            obs = next_obs
            total_reward += reward

        epsilon = max(AGENT_CONFIG["epsilon_end"], AGENT_CONFIG["epsilon_decay"] * epsilon)
        total_rewards.append(total_reward)

    env.close()
    score = np.mean(total_rewards[-5:])

    # --- 3. Memory Management ---
    del env, agent
    gc.collect()

    return score

if __name__ == "__main__":
    results_dir = "sumo_d3qn/tuning_results"
    os.makedirs(results_dir, exist_ok=True)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=40) # Set for ~9-10 hour run

    print("\n--- Optuna Study Complete ---")
    print(f"Number of finished trials: {len(study.trials)}")
    print("Best trial:")
    trial = study.best_trial

    print(f"  Value (Score): {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    # --- 4. Generate and Save Plots ---
    try:
        history_fig = optuna.visualization.plot_optimization_history(study)
        history_fig.write_html(os.path.join(results_dir, "optimization_history.html"))

        importance_fig = optuna.visualization.plot_param_importances(study)
        importance_fig.write_html(os.path.join(results_dir, "param_importances.html"))

        print(f"\nSuccessfully saved study plots to '{results_dir}'")
        print("You can open the .html files in your browser to view the interactive plots.")

    except (ImportError, ModuleNotFoundError):
        print(f"\nCould not generate plots. Please install plotly: pip install plotly")
    except Exception as e:
        print(f"\nAn error occurred during plot generation: {e}")

    # --- 5. Save Best Hyperparameters ---
    best_params_path = os.path.join(results_dir, "best_hyperparameters.txt")
    with open(best_params_path, "w") as f:
        f.write("Best Hyperparameters Found by Optuna:\n")
        for key, value in trial.params.items():
            f.write(f"{key}: {value}\n")
        f.write(f"\nBest Score (Value): {trial.value}\n")
    print(f"Best hyperparameters saved to '{best_params_path}'")