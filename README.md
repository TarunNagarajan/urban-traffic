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

## Advanced Statistics and Visualization

For a more in-depth analysis of the agent's performance, a `plot_statistics.py` script is provided. This script generates advanced comparison plots beyond the default waiting time and queue length.

### Workflow

1.  **Run Evaluation:** First, run the evaluation script to generate the necessary data files.
    ```bash
    python evaluation.py
    ```
    This will create several files in the `logs/evaluation` directory, including `d3qn_rewards.csv`, `baseline_rewards.csv`, and `emissions.xml`.

2.  **Generate Plots:** Next, run the `plot_statistics.py` script.
    ```bash
    python plot_statistics.py
    ```
    This will read the data from the `logs/evaluation` directory and generate new comparison plots, such as the rolling reward comparison.

### How to Add a New Statistic/Plot

The `plot_statistics.py` script is designed to be easily extensible. Here is a guide on how to add a new plot, using "Average Speed" as an example.

1.  **Open `plot_statistics.py`:** Open the script in your code editor.

2.  **Create a New Plotting Function:** Add a new Python function that takes the necessary data files as input. For average speed, we need the main simulation output files.

    ```python
    def plot_average_speed(rl_sim_csv, baseline_sim_csv, output_dir):
        """
        Plots the average system speed for both the D3QN agent and the baseline.
        """
        # Step 1: Load the data from the CSV files
        # The separator is a semicolon ';'
        rl_df = pd.read_csv(rl_sim_csv, sep=';')
        baseline_df = pd.read_csv(baseline_sim_csv, sep=';')

        # Step 2: Create the plot
        plt.figure(figsize=(12, 6))
        
        # Plot the 'system_mean_speed' column for both dataframes
        plt.plot(rl_df['step'], rl_df['system_mean_speed'], label='D3QN Agent')
        plt.plot(baseline_df['step'], baseline_df['system_mean_speed'], label='Fixed-Time Baseline')
        
        # Step 3: Add labels and title
        plt.xlabel('Simulation Step')
        plt.ylabel('Average System Speed (m/s)')
        plt.title('D3QN Agent vs. Fixed-Time Baseline: Average Speed')
        plt.legend()
        plt.grid(True)
        
        # Step 4: Save the plot
        plot_path = os.path.join(output_dir, 'average_speed_comparison.png')
        plt.savefig(plot_path)
        print(f"Saved average speed plot to {plot_path}")
        plt.close()
    ```

3.  **Call the New Function:** In the `if __name__ == "__main__":` block at the bottom of the script, find the placeholder for your new plot and replace it with a call to your new function.

    **Before:**
    ```python
    # --- Plot Average Speed (Placeholder) ---
    plot_average_speed(rl_sim_csv, baseline_sim_csv, eval_log_dir)
    ```

    **After:**
    ```python
    # --- Plot Average Speed ---
    plot_average_speed(rl_sim_csv, baseline_sim_csv, eval_log_dir)
    ```
    (You would also remove the placeholder implementation from the function itself).

You can follow this same pattern to implement other plots, such as for fuel consumption by parsing the `emissions.xml` file.