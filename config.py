import torch

SUMO_CONFIG = {
    # Simulation settings
    "sumo_cfg_file": "path/to/your/sumo.sumocfg",  # Placeholder for SUMO config file
    "net_file": "path/to/your/net.net.xml", # Placeholder for the network file from OpenStreetMap
    "route_file": "path/to/your/routes.rou.xml", # Placeholder for the route file
    "num_seconds": 900,  # Total simulation time in seconds (e.g., 15 minutes)
    "delta_time": 5,  # Time step for agent action (in seconds)
    "yellow_time": 2,  # Duration of yellow light phase
    "min_green": 10,  # Minimum green time for a phase
    "gui": True,  # Set to True to run with SUMO GUI
    
    # Vehicle classes
    "vehicle_classes": {
        "car": {"vClass": "passenger", "speedDev": 0.1, "laneChangeModel": "LC2013"},
        "motorcycle": {"vClass": "motorcycle", "speedDev": 0.2},
        "bus": {"vClass": "bus", "speedDev": 0.1},
        "bicycle": {"vClass": "bicycle", "speedDev": 0.3},
        "pedestrian": {"vClass": "pedestrian", "speedDev": 0.1} # Note: Pedestrian modeling is more complex
    }
}

AGENT_CONFIG = {
    # D3QN Agent parameters
    "state_size": 0,  # To be dynamically set based on environment
    "action_size": 2,  # 0: HOLD_PHASE, 1: SWITCH_PHASE
    "buffer_size": int(1e5),  # Replay buffer size
    "batch_size": 64,  # Minibatch size
    "gamma": 0.9958,  # Discount factor
    "lr": 0.0003,  # Learning rate
    "tau": 1e-3,  # For soft update of target parameters
    "update_every": 4,  # How often to update the network
    "epsilon_start": 1.0,  # Starting value of epsilon
    "epsilon_end": 0.01,  # Minimum value of epsilon
    "epsilon_decay": 0.995,  # Epsilon decay rate
    "device": "cuda" if torch.cuda.is_available() else "cpu",

    # Dueling DQN Network
    "fc1_units": 128,
    "fc2_units": 128,
}

TRAINING_CONFIG = {
    # Training loop settings
    "episodes": 360,  # Number of training episodes
    "save_checkpoint_every": 10, # How often to save a checkpoint
    "max_steps": SUMO_CONFIG["num_seconds"] // SUMO_CONFIG["delta_time"], # Max steps per episode
    "log_dir": "logs/", # Directory for logging metrics
    "model_save_path": "models/d3qn_agent.pth", # Path to save the best model
}

REWARD_CONFIG = {
    # Weights for the reward function components
    "use_queue_length": True,
    "queue_length_weight": -0.1,
    "use_jerk_penalty": True,
    "jerk_penalty": -0.329,
    "use_virtual_pedestrian_penalty": True,
    "virtual_pedestrian_penalty": -0.2,
    "use_fuel_consumption_penalty": True,
    "fuel_consumption_penalty": -0.1,
    "use_pressure": True,
    "pressure_weight": -0.01,
    "use_throughput": True,
    "throughput_weight": 1.0,
    "use_waiting_time_penalty": False,
    "waiting_time_penalty": -0.1,
    "use_stops_penalty": False,
    "stops_penalty": -0.5,
    "use_fairness_penalty": False,
    "fairness_penalty": -0.3,
    "use_spillback_penalty": False,
    "spillback_penalty": -10.0,
    "spillback_threshold": 50,
}

# Placeholder for GRU-based inflow prediction
# For now, we'll use random values as a stub.
STUB_GRU_PREDICTION = {
    "use_stub": True,
    "inflow_dimension": 4 # Example: inflow from 4 directions
}

TUNING_CONFIG = {
    "use_switching_cooldown": True,
    "switching_cooldown_seconds": 20,
    "use_queue_override": True,
    "queue_override_threshold": 30,

    "use_adaptive_thresholds": True,
    "density_regimes": [150, 300], 
    "queue_override_thresholds": [40, 30, 20],

    "use_phase_commitment": True,
    "phase_commitment_steps": 2,

    "use_temperature_sampling": True,
    "inference_temperature": 0.5
}