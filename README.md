# Urban Traffic Control with Deep Reinforcement Learning

This project utilizes Deep Reinforcement Learning to manage traffic flow in a simulated urban environment. It includes two types of agents: a D3QN (Dueling Double Deep Q-Network) and a PPO (Proximal Policy Optimization) agent. The system is designed to control multiple intersections in a coordinated, communication-enabled manner.

## Performance Improvements

Through a systematic process of analysis and inference-time tuning, we were able to significantly improve the performance of the base D3QN agent **without any retraining**.

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

### 1. SUMO Installation

SUMO (Simulation of Urban MObility) is required to run the traffic simulations.

1.  **Download SUMO:** Go to the official SUMO download page: [https://sumo.dlr.de/docs/Downloads.php](https://sumo.dlr.de/docs/Downloads.php)
2.  **Install:** Download and install a recent, stable version for Windows. The installer will guide you through the process.

### 2. Environment Variable Setup

After installing SUMO, you must add it to your system's environment variables to allow the Python scripts to find and use it.

1.  **Open System Properties:** Press `Win + R`, type `sysdm.cpl`, and press Enter.
2.  **Go to Environment Variables:** In the System Properties window, go to the `Advanced` tab and click on the `Environment Variables...` button.
3.  **Create `SUMO_HOME`:**
    *   Under `System variables`, click `New...`.
    *   For `Variable name:`, enter `SUMO_HOME`.
    *   For `Variable value:`, enter the path to your SUMO installation directory. This is typically `C:\Program Files (x86)\Eclipse\Sumo`.
4.  **Add SUMO to the `Path` variable:**
    *   Under `System variables`, find the `Path` variable, select it, and click `Edit...`.
    *   Click `New` and add a new entry: `%SUMO_HOME%\bin`.
    *   Click `New` again and add another entry: `%SUMO_HOME%\tools`.
    *   Click `OK` to close all windows.

### 3. Project Dependencies

This project requires several Python packages. You can install them using `pip` and the provided `requirements.txt` file.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/TarunNagarajan/urban-traffic.git
    cd urban-traffic
    ```
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### D3QN Agent

#### Training
To train the D3QN agent, use `train.py`. You can resume from a checkpoint and set the total number of episodes.

**Example: Fine-tuning from episode 430 for 200 more episodes:**
```bash
python train.py --checkpoint_path logs/20250926-134923/checkpoint_ep430.pth --start_episode 431 --episodes 630
```

#### Evaluation
To evaluate a specific D3QN agent checkpoint, use `evaluation.py`. The script will run the agent and a baseline, then generate comparison plots and performance breakdowns.

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
- **`use_temperature_sampling`**: (boolean) If true, uses softmax sampling for actions instead of always picking the best one.
- **`inference_temperature`**: (float) The temperature for sampling. Higher values lead to more random actions.

## Understanding the Evaluation Output

The `evaluation.py` script generates a set of files in the `logs/evaluation/` directory, giving a comprehensive view of performance.

### High-Level Summary (`..._results.csv`)

These files provide a summary of the entire system's performance at each second of the simulation.

| Column                      | Explanation                                                                                      |
| --------------------------- | ------------------------------------------------------------------------------------------------ |
| `step`                      | The timestamp of the simulation in seconds.                                                      |
| `system_total_waiting_time` | The combined waiting time (in seconds) of all vehicles up to this point. A key measure of congestion. |
| `system_mean_queue_length`  | The average number of cars waiting at all traffic lights. A key indicator of traffic flow.      |
| `system_total_stopped`      | The total number of stopped vehicles across the map at that moment.                                |
| `system_mean_speed`         | The average speed of all cars on the map at this moment (in meters/second).                      |

### Step-by-Step Details (`..._details.csv`)

These files give a granular, step-by-step log of the agent's decisions and their immediate consequences.

| Column                | Explanation                                                                                             |
| --------------------- | ------------------------------------------------------------------------------------------------------- |
| `step`                | The timestamp of the simulation in seconds.                                                             |
| `total_reward`        | The overall reward score for the system at this step, based on the components in `REWARD_CONFIG`.       |
| `{ts_id}_phase`       | Which direction of traffic has the green light for a specific intersection (`ts_id`).                     |
| `{ts_id}_total_queue` | The number of cars waiting in line at that specific intersection.                                         |

*(The `{ts_id}` will be replaced by the actual ID of each traffic light in the network, e.g., `gneJ0_phase`)*

### Environmental Impact (`emissions.xml`)

This file contains detailed environmental data, logging the fuel consumption and CO2 emissions for every single vehicle at every step of the simulation. It's the source for any analysis of the agent's environmental impact.
