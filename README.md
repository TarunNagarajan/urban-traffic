# Urban Traffic Control with Deep Reinforcement Learning

This project utilizes a Deep Reinforcement Learning agent to manage traffic flow in a simulated urban environment. The agent is built using a Dueling Double Deep Q-Network (D3QN) and is designed to control multiple intersections in a coordinated, communication-enabled manner.

## Features

- **Multi-Agent Control:** The system controls all traffic lights in the provided network, not just a single intersection.
- **Communication-Enabled Agents:** Agents communicate by sharing state information with their immediate neighbors, allowing for more coordinated and intelligent traffic management.
- **Generalizable Architecture:** The solution is designed to be flexible and can be applied to any road network topology defined in a SUMO `.net.xml` file.
- **D3QN Agent:** Employs a Dueling Double Deep Q-Network for efficient learning and decision-making.
- **Hyperparameter Tuning:** Includes a script to automatically tune the agent's hyperparameters using Optuna.

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

### Training

To train a new agent, run the `train.py` script. The script will train the agent for the number of episodes specified in `config.py` and save the best performing model to `models/d3qn_agent.pth`.

```bash
python train.py
```

### Evaluation

To evaluate a trained agent against the default fixed-time controller, run the `evaluation.py` script. This requires a trained model to be present in the `models` directory.

```bash
python evaluation.py
```

### Hyperparameter Tuning

To find the best hyperparameters for the agent, run the `tune.py` script. This will run a series of training sessions to find the optimal learning rate, reward weights, etc.

```bash
python tune.py
```

## Understanding the Output

### Training Output

During training, you will see output for each episode:

**Example:**
```
Episode 1/100 | Total Reward: -12345.67 | Epsilon: 0.9950
```

- **Total Reward:** This is the cumulative reward for the episode. Since the reward is based on negative queue length, a higher (less negative) number is better.
- **Epsilon:** This is the exploration rate of the agent. It will decrease over time as the agent learns.

When a new best model is found (i.e., a model that achieves a higher total reward than any previous episode), it will be saved, and you will see the message:
`New best model saved with reward: -12345.67`

### Evaluation Output

After running the evaluation, two comparison plots will be saved in the `logs/evaluation` directory:

1.  **`waiting_time_comparison.png`**: This plot compares the total system waiting time (in seconds) of your D3QN agent against the fixed-time baseline. **Lower values are better**, indicating that vehicles spend less time waiting at intersections.
2.  **`queue_length_comparison.png`**: This plot compares the average queue length (in number of vehicles) across all intersections. **Lower values are better**, indicating less congestion.

Two CSV files, `d3qn_results.csv` and `baseline_results.csv`, are also generated with detailed simulation data.

### Tuning Output

During hyperparameter tuning, Optuna will print logs for each trial it completes.

**Example:**
```
[I 2025-09-26 12:00:00,000] Trial 0 finished with value: -25000.0 and parameters: {'queue_length_weight': -0.5, 'jerk_penalty': -0.2, ...}. Best is trial 0 with value: -25000.0.
```

- **Value:** This is the score for the trial (the average reward over the last few episodes). A higher (less negative) value is better.
- **Parameters:** The set of hyperparameters used for that trial.

After the tuning is complete, you will find the following in the `sumo_d3qn/tuning_results` directory:

1.  **`best_hyperparameters.txt`**: A text file containing the best set of hyperparameters found.
2.  **`optimization_history.html`**: An interactive plot showing the progress of the optimization. You can open this in your browser to see how the score improved over time.
3.  **`param_importances.html`**: An interactive plot showing which hyperparameters had the most impact on the final score.
