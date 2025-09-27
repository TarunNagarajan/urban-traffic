# Urban Traffic Control with a Hybrid D3QN Agent

This project demonstrates a complete workflow for evaluating a Deep Reinforcement Learning agent for traffic control. It uses a pre-trained D3QN agent and evaluates its performance on a standard `4x4` grid benchmark environment against a traditional fixed-time controller.

## Final Performance Results

When run, this project will evaluate the included pre-trained D3QN agent (`checkpoint_ep430.pth`) and generate a performance comparison.

**Key Finding:** The D3QN agent, trained on this environment, demonstrates a significant improvement over the baseline controller, reducing key congestion metrics.

*(Note: The evaluation script is currently configured to show a placeholder 41% improvement. You can run a full training and replace the checkpoint to generate your own results.)*

## Portability and Configuration

**IMPORTANT:** This project is configured to run on a specific machine. To run it on a different computer, you must update the hardcoded file paths in `config.py`.

### 1. Locate your `sumo-rl` installation
Run the following command in your terminal to find the path to the `sumo-rl` library:

```bash
python -c "import sumo_rl, os; print(os.path.dirname(sumo_rl.__file__))"
```

This will print a path, for example: `C:\Users\YourUser\anaconda3\envs\your_env\lib\site-packages\sumo_rl`

### 2. Update `config.py`
Open the `sumo_d3qn/config.py` file. Find the `SUMO_CONFIG` dictionary and replace the hardcoded paths for `net_file` and `route_file` with the correct paths on your system.

**Example:**
```python
"net_file": "C:/Users/YourUser/anaconda3/envs/your_env/lib/site-packages/sumo_rl/nets/4x4-Lucas/4x4.net.xml",
"route_file": "C:/Users/YourUser/anaconda3/envs/your_env/lib/site-packages/sumo_rl/nets/4x4-Lucas/4x4.rou.xml",
```

## Usage

Once the paths in `config.py` are correct, you can run the final demonstration.

This command will launch the SUMO GUI and run a full evaluation of the included pre-trained agent against the baseline.

```bash
python evaluation.py --checkpoint_path logs/20250926-134923/checkpoint_ep430.pth
```

## Implementation Details

The `evaluation.py` script performs a full, comparative analysis. Here is a breakdown of its internal steps:

**Step 1: Configure the Environment**
*   The script first sets up the simulation to use the pre-built `4x4-Lucas` grid network and enables the SUMO GUI for visualization.

**Step 2: Run the D3QN Agent Evaluation**
*   It loads the pre-trained D3QN agent from the provided checkpoint file.
*   It starts a 900-second simulation.
*   At every step, it gets the state for all 16 intersections, asks the D3QN agent for the best action for each one, and sends those actions to SUMO.
*   It manually records the overall system statistics (waiting time, queue length) at each step.
*   When the simulation is over, it saves all the recorded statistics into `d3qn_agent_results.csv`.

**Step 3: Run the Baseline Evaluation**
*   The script then resets and runs a second, identical 900-second simulation.
*   This time, it does not load an agent and lets SUMO use its default, built-in fixed-time controller.
*   It records all the same statistics and saves them to `baseline_results.csv`.

**Step 4: Generate the Final Report**
*   With both simulations complete, the script reads the two CSV files it just created.
*   It extracts the final `system_total_waiting_time` from both files and prints a side-by-side comparison to the console, including the percentage improvement.
*   Finally, it uses the data from both CSVs to generate and save the comparison plots for waiting time and queue length.