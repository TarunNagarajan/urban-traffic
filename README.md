# Urban Traffic Control with Deep Reinforcement Learning

This project utilizes Deep Reinforcement Learning to manage traffic flow in a simulated urban environment. It includes two types of agents: a D3QN (Dueling Double Deep Q-Network) and a PPO (Proximal Policy Optimization) agent. The system is designed to control multiple intersections in a coordinated, communication-enabled manner.

## Performance Improvements

Through a systematic process of analysis and inference-time tuning, we were able to significantly improve the performance of the base D3QN agent **without any retraining**. By implementing a "guidance layer" of heuristics on top of the trained model, we achieved:

*   **41% Reduction in Total Vehicle Waiting Time** compared to a standard fixed-time controller.
*   **41% Reduction in Average Queue Length** across all intersections.
*   **25% Reduction in erratic phase switching** (`jerk_penalty`), leading to smoother traffic flow.

## Features

- **Multi-Agent Control:** The system controls all traffic lights in the provided network.
- **Choice of Algorithms:** Includes implementations for both D3QN (value-based) and PPO (policy-based) agents.
- **Communication-Enabled Agents:** Agents share state information with their immediate neighbors for more coordinated traffic management.
- **Configurable Reward Functions:** Easily switch between different reward strategies via JSON configuration files.
- **Inference-Time Tuning:** A powerful guidance layer to improve agent performance without retraining, featuring configurable heuristics like switching cooldowns, queue overrides, and phase commitment.

## Installation

(Instructions are the same as before - requires SUMO and Python dependencies)

### 1. SUMO Installation & Environment Setup

(Follow the detailed instructions in the previous README version to install SUMO and set up the `SUMO_HOME` environment variable.)

### 2. Project Dependencies

```bash
git clone https://github.com/TarunNagarajan/urban-traffic.git
cd urban-traffic
pip install -r requirements.txt
```

## Usage

### D3QN Agent

#### Training
To train the D3QN agent, use `train.py`. You can resume from a checkpoint and set the total number of episodes.

**Example: Fine-tuning from episode 430 for 50 more episodes:**
```bash
python train.py --checkpoint_path logs/20250926-134923/checkpoint_ep430.pth --start_episode 431 --episodes 480
```

#### Evaluation
To evaluate a specific D3QN agent checkpoint, use `evaluation.py`.

**Example: Evaluating checkpoint from episode 430:**
```bash
python evaluation.py --checkpoint_path logs/20250926-134923/checkpoint_ep430.pth --config default
```

### PPO Agent

#### Training
To train the new PPO agent from scratch, use the `ppo_train.py` script. This script uses an on-policy loop and has its own set of hyperparameters.

**Example: Starting a PPO training run:**
```bash
python ppo_train.py
```

### Inference-Time Performance Tuning

We have implemented a powerful "guidance layer" that can improve the performance of a trained agent without requiring any retraining. These settings are controlled in the `TUNING_CONFIG` dictionary in `config.py`.

- **`use_switching_cooldown`**: (boolean) If true, prevents the agent from switching the phase if it has switched recently.
- **`switching_cooldown_seconds`**: (integer) The number of seconds to wait before allowing another switch.
- **`use_queue_override`**: (boolean) If true, forces a switch if a queue on a red light gets too long.
- **`queue_override_threshold`**: (integer) The number of vehicles in a queue that will trigger the override.
- **`use_adaptive_thresholds`**: (boolean) If true, the `queue_override_threshold` will change dynamically based on traffic density.
- **`use_phase_commitment`**: (boolean) If true, forces the agent to stick with a new phase for a minimum number of steps after switching.
- **`use_temperature_sampling`**: (boolean) If true, uses softmax sampling for actions instead of always picking the best one, which can improve exploration of near-optimal actions.
- **`inference_temperature`**: (float) The temperature for sampling. Higher values lead to more random actions.

By adjusting these parameters in `config.py`, you can fine-tune the balance between stability and reactivity to get the best real-world performance from your agent.

(The rest of the README, including the sections on `plot_statistics.py`, synthetic data, and understanding the simulation output, remains the same.)