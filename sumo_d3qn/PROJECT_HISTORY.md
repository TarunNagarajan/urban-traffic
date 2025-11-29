# Project History and Context: D3QN SUMO Traffic Control

This document provides a comprehensive history of the development and decision-making process for the `D3QN_sumo` project. Its purpose is to serve as a detailed context file for any developer or AI assistant continuing the work.

## Phase 1: Initial Exploration and Bug Fixing

### 1.1. Initial Project State

The project started with a set of Python scripts for training a D3QN agent to control a single traffic light in a SUMO simulation. The core files were:

-   `agent.py`: Defined the D3QN agent, Q-Network, and Replay Buffer.
-   `config.py`: Contained configuration for the agent, simulation, and training.
-   `train.py`: The main script for training the agent.
-   `evaluation.py`: A script to evaluate the trained agent against a baseline.
-   `tune.py`: A script for hyperparameter tuning using Optuna.

### 1.2. First Evaluation and `state_size` Mismatch Bug

**Action:** The first task was to run the evaluation script (`evaluation.py`) to understand the project's baseline performance.

**Problem:** The script failed with a `RuntimeError: size mismatch for fc1.weight`. The error indicated that the model being loaded from the checkpoint had a different input size (10) than the model being created in the evaluation script (6).

**Analysis:**
-   The model trained via `train.py` used a `state_size` calculated with a `gru_stub_dim` of 4 (from `STUB_GRU_PREDICTION["inflow_dimension"]` in `config.py`). The state was composed of: `queue_length` (4) + `phase` (2) + `gru_stub` (4) = **10**.
-   The `evaluation.py` script was hardcoded to use a `gru_stub_dim` of 0, resulting in a `state_size` of: `queue_length` (4) + `phase` (2) = **6**.

**Solution:** The `evaluation.py` script was modified to use the same `inflow_dimension` from the `config.py` file, ensuring that the model architecture was consistent between training and evaluation.

## Phase 2: Implementing a Generalizable Multi-Agent System

### 2.1. User Request

The user requested a major architectural change: to move from a single-agent system to a multi-agent system that could:
1.  Control all intersections in the network.
2.  Enable "communication" between the agents.
3.  Be generalizable to different road network layouts (e.g., a 3-intersection triangle), not just the 4x4 grid.

### 2.2. The "Train-General, Test-Specific" Methodology

Before implementation, a key strategic decision was made:

-   **Training:** Should continue on the generic 4x4 grid. This forces the agent to learn more generalizable policies and avoids overfitting to a specific, simple network.
-   **Evaluation:** Should be performed on the specific, real-world map the user cares about (e.g., the Bhubaneswar map). This provides a more powerful demonstration of the agent's ability to adapt to a new environment.
-   This approach was approved by the user.

### 2.3. Implementation of Multi-Agent Communication

**Solution Proposed:**
-   Use a single, shared `D3QNAgent` for all traffic lights (parameter sharing).
-   Implement "communication" by enhancing the state representation of each agent to include information (queue lengths and phase) from its immediate neighbors.
-   Use a global reward function based on the total system-wide queue length.

**Initial Implementation:**
-   The `train.py`, `evaluation.py`, and `tune.py` scripts were modified to loop over all traffic signal IDs (`ts_ids`) at each step.
-   A new `compute_state` function was created to concatenate the local state with the states of its neighbors. To handle varying numbers of neighbors, the state vector was padded with zeros up to a `max_neighbors` count.

### 2.4. The Neighbor-Finding Bug Saga

This phase was marked by a series of difficult-to-debug errors related to finding the neighbors of a traffic light within the `sumo-rl` and `sumolib` libraries. This was the most challenging part of the project.

-   **Attempt 1:** `env.traffic_signals[ts_id].neighbors`
    -   **Error:** `AttributeError: 'TrafficSignal' object has no attribute 'neighbors'`
    -   **Reason:** This was an incorrect assumption about the structure of the `TrafficSignal` class in `sumo-rl`.

-   **Attempt 2:** `env.net`
    -   **Error:** `AttributeError: 'SumoEnvironment' object has no attribute 'net'`
    -   **Reason:** The parsed `sumolib` network object is not stored as a public `net` attribute on the environment.

-   **Attempt 3:** `env._net`
    -   **Error:** `AttributeError: 'str' object has no attribute 'getTLS'`
    -   **Reason:** `env._net` was discovered to be just a *string* containing the file path to the `.net.xml` file, not the parsed network object itself.

-   **Subsequent `sumolib` Failures:** After manually loading the network with `net = sumolib.net.readNet(env._net)`, further attempts failed due to what appeared to be `sumolib` API version inconsistencies.
    -   `'TLS' object has no attribute 'getNeighbors'`
    -   `'TLS' object has no attribute 'getNode'` or `'_node'`
    -   `'Net' object has no attribute 'getNeighboringTLS'`

-   **Final Solution (Provided by the User):** The user provided a robust, correct function (`get_neighboring_traffic_lights`) that correctly traverses the network graph to find neighbors. This function was integrated into all scripts, finally resolving the issue.

## Phase 3: Documentation and Usability

### 3.1. `requirements.txt`

A `requirements.txt` file was created to list all project dependencies (`torch`, `numpy`, `pandas`, `matplotlib`, `sumo-rl`, `optuna`).

### 3.2. `README.md` Creation and Refinement

-   A comprehensive `README.md` was created, including:
    -   Installation instructions for SUMO and Python dependencies.
    -   Detailed setup guide for environment variables (`SUMO_HOME`).
    -   Usage instructions for all scripts (`train`, `evaluate`, `tune`).
    -   A detailed breakdown of the format of all output files (`d3qn_results.csv`, `d3qn_details.csv`, `emissions.xml`).
    -   An explanation of the real-time SUMO simulation output.
-   A new script, `plot_statistics.py`, was created to provide a framework for advanced analysis.
-   The `README.md` was updated with a new section, "Advanced Statistics and Visualization," which included a step-by-step tutorial on how a developer could add new plots, using "Average Speed" as a concrete example.
-   A section on "Developing with Synthetic Data" was added to explain how a developer could work on the plotting scripts without waiting for simulations to finish.

### 3.3. Synthetic Data Generation

To facilitate parallel development, a full set of synthetic data files was generated and committed to the repository in `logs/evaluation/`. This allows a developer to immediately run `plot_statistics.py` and build out the dashboard and analysis features.

### 3.4. Git Repository Management

The project was initialized as a Git repository and pushed to the user-provided GitHub URL (`https://github.com/TarunNagarajan/urban-traffic.git`). This involved:
-   Initializing the repository (`git init`).
-   Adding the remote URL.
-   Staging, committing, and pushing all project files, including the new documentation and synthetic data.
-   Troubleshooting several issues related to shell quoting for commit messages and force-pushing to an initialized remote repository.

## Phase 4: Advanced Reward Functions and Agent Logic

### 4.1. Virtual Pedestrian Penalty

-   **Request:** The user wanted to incorporate a penalty for pedestrian waiting time, but without a full pedestrian simulation.
-   **Implementation:** A "virtual" penalty was added. The vehicle queue length is used as a heuristic for pedestrian waiting time. A new `virtual_pedestrian_penalty` was added to `REWARD_CONFIG` and incorporated into the `compute_reward` function.
-   **Constraint:** This was implemented *after* a long tuning run had started, so this new reward weight was deliberately excluded from the `tune.py` script for that run.

### 4.2. Fuel Consumption Penalty

-   **Request:** The user wanted to add a penalty based on fuel consumption, as seen in the reference article.
-   **Implementation:** The `compute_reward` function was modified to accept the `env` object. At each step, it now iterates through all active vehicles using `env.sumo.vehicle.getIDList()` and sums their fuel consumption using `env.sumo.vehicle.getFuelConsumption(veh_id)`. A new `fuel_consumption_penalty` was added to `REWARD_CONFIG`.

### 4.3. Action Masking ("Cooldown")

-   **Request:** The user wanted to properly implement the "cooldown" or minimum green time mentioned in the reference article.
-   **Implementation:** A sophisticated **Action Masking** technique was implemented.
    1.  The `act` method in `agent.py` was upgraded to accept an `action_mask`.
    2.  This mask is used to apply a large negative value to the Q-value of invalid actions (e.g., switching the light before `min_green` time has passed), preventing the agent from ever choosing them.
    3.  The main loops in `train.py` and `evaluation.py` were updated to check `env.traffic_signals[ts_id].time_since_last_phase_change` at every step and pass the correct mask to the agent.
    4.  A `min_green` parameter was added to `SUMO_CONFIG`.

## Current Project State

-   The project is a robust, generalizable, multi-agent D3QN system.
-   The reward function is comprehensive, including penalties for queue length, phase switching, virtual pedestrians, and fuel consumption.
-   The agent's intelligence is enhanced with Action Masking to respect traffic light cooldowns.
-   The project is thoroughly documented in the `README.md`.
-   The codebase is version-controlled with Git and pushed to the user's repository.
-   A hyperparameter tuning run is currently in progress by the user.

## Future Work (Drafted GitHub Issues)

### Issue 1: Integrate YOLOv8 for Real-time Pedestrian Detection

-   **Goal:** Replace the virtual pedestrian heuristic with real data from a YOLOv8 model processing simulated camera feeds.
-   **Workflow:** Setup YOLOv8 -> Extract pedestrian counts -> Create an interface to pass data to the Python script (e.g., a JSON file) -> Modify `compute_state` to include pedestrian counts -> Modify `compute_reward` to use real data.
-   **Impact:** This would require retraining the model due to the change in `state_size`.

### Issue 2: Design and Metrics for Performance Visualization Dashboard

-   **Goal:** Create a real-time dashboard for visualizing simulation performance.
-   **Components:** Global System-Wide Metrics, Intersection-Specific View, Agent-Specific Information.
-   **Metrics to Stream:** `system_total_waiting_time`, `system_mean_speed`, `total_fuel_consumption`, `global_reward`, per-intersection queue/phase data, agent's `epsilon`, and Q-values.
-   **Suggested Tech:** A new script (`run_dashboard.py`) to generate a JSON file at each step, which can be read by a dashboard built with a framework like Dash or Streamlit.
