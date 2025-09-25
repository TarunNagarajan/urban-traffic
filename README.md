# Urban Traffic Control with Deep Reinforcement Learning

This project utilizes a Deep Reinforcement Learning agent to manage traffic flow in a simulated urban environment. The agent is built using a Dueling Double Deep Q-Network (D3QN) and is designed to control multiple intersections in a coordinated, communication-enabled manner.

## Features

- **Multi-Agent Control:** The system controls all traffic lights in the provided network, not just a single intersection.
- **Communication-Enabled Agents:** Agents communicate by sharing state information with their immediate neighbors, allowing for more coordinated and intelligent traffic management.
- **Generalizable Architecture:** The solution is designed to be flexible and can be applied to any road network topology defined in a SUMO `.net.xml` file.
- **D3QN Agent:** Employs a Dueling Double Deep Q-Network for efficient learning and decision-making.
- **Hyperparameter Tuning:** Includes a script to automatically tune the agent's hyperparameters using Optuna.
- **Advanced Statistics:** Provides a framework for detailed performance analysis, including statistics on fuel consumption, waiting times, and rewards.

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

To train a new agent, run the `train.py` script.

```bash
python train.py
```

### Evaluation & Statistics

1.  **Run Evaluation:** This generates the raw data from the simulation.
    ```bash
    python evaluation.py
    ```
2.  **Generate Plots and Statistics:** This processes the raw data.
    ```bash
    python plot_statistics.py
    ```

### Hyperparameter Tuning

To find the best hyperparameters for the agent, run the `tune.py` script.

```bash
python tune.py
```

## Understanding the Simulation Output

When you run any of the scripts that interact with SUMO (`train.py`, `evaluation.py`, `tune.py`), you will see a continuous stream of output from the SUMO engine itself. This provides real-time information about the simulation's performance.

**Example Output Line:**
```
Step #1800.00 (3ms ~= 333.33*RT, ~89666.67UPS, TraCI: 187ms, vehicles TOT 2665 ACT 261 BUF
```

Here is a breakdown of what each part means:

-   `Step #1800.00`: The current time in the simulation, in seconds.
-   `(3ms ~= 333.33*RT)`:
    -   `3ms`: The real-world time it took your computer to process this single simulation step.
    -   `333.33*RT`: The real-time factor. This indicates that the simulation is currently running approximately 333 times faster than real life. A higher number is better.
-   `~89666.67UPS`: "Updates Per Second". This is another measure of simulation speed, representing the number of vehicle updates the engine can perform per second.
-   `TraCI: 187ms`: The time spent communicating with the TraCI (Traffic Control Interface). This is the overhead for the Python script to send commands to and receive data from SUMO.
-   `vehicles TOT 2665 ACT 261 BUF`:
    -   `TOT 2665`: The total number of vehicles that have entered the simulation since it began.
    -   `ACT 261`: The number of "active" vehicles currently driving in the network.
    -   `BUF`: The number of vehicles currently in a buffer, waiting to be inserted into the network at their scheduled departure time.

## Data and Output Files Format

The evaluation process generates several files in the `logs/evaluation/` directory. These files are designed to be consumed by the `plot_statistics.py` script and for your own custom analysis.

### `d3qn_results.csv` / `baseline_results.csv`

These files are generated by the `sumo-rl` library and contain high-level, system-wide metrics for each step of the simulation.

| Column                      | Description                                                                 |
| --------------------------- | --------------------------------------------------------------------------- |
| `step`                      | The simulation step number.                                                 |
| `system_total_stopped`      | Total number of vehicles stopped in the entire network.                     |
| `system_total_waiting_time` | Sum of the waiting time for all vehicles in the network.                    |
| `system_mean_waiting_time`  | Average waiting time per vehicle in the network.                            |
| `system_mean_speed`         | Average speed of all vehicles in the network (m/s).                         |
| `system_mean_queue_length`  | Average queue length over all lanes in the network.                         |
| ...                         | Other system-level metrics.                                                 |

### `d3qn_details.csv` / `baseline_details.csv`

These files provide more granular, step-by-step data that is collected during the evaluation run. This is the primary source of data for custom plots and statistics.

| Column                | Description                                                              |
| --------------------- | ------------------------------------------------------------------------ |
| `step`                | The simulation step number.                                              |
| `reward`              | The global reward for the entire system at this step.                    |
| `{ts_id}_phase`       | The integer index of the active green phase for a specific traffic light. |
| `{ts_id}_total_queue` | The total number of vehicles queued at a specific traffic light.         |

*(The `{ts_id}` will be replaced by the actual ID of each traffic light in the network, e.g., `gneJ0_phase`)*

### `emissions.xml`

This XML file contains detailed emission data for every vehicle at every step of the simulation. It is useful for calculating fuel consumption and other environmental metrics.

**Structure:**
```xml
<emission-export>
    <timestep time="0.00">
        <vehicle id="..." CO2="..." fuel="..." />
        <vehicle id="..." CO2="..." fuel="..." />
        ...
    </timestep>
    <timestep time="1.00">
        ...
    </timestep>
</emission-export>
```

## Advanced Statistics and Visualization

The `plot_statistics.py` script is provided as a starting point for in-depth analysis. It computes summary statistics and generates a rolling reward plot. It is designed to be easily extended by your team.

### How to Add a New Statistic/Plot

Here is a guide on how to add a new plot, using "Average Speed" as an example.

1.  **Open `plot_statistics.py`:** Open the script in your code editor.

2.  **Create a New Plotting Function:** Add a new Python function that takes the necessary data files as input. For average speed, we need the main simulation output files (`d3qn_results.csv` and `baseline_results.csv`).

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

You can follow this same pattern to implement other plots and statistical computations.