import os
import sys
import numpy as np
import torch
import sumo_rl
import sumolib
from datetime import datetime

from agent import D3QNAgent
from config import SUMO_CONFIG, AGENT_CONFIG, TRAINING_CONFIG, REWARD_CONFIG, STUB_GRU_PREDICTION

# --- Constants based on environment inspection ---
NUM_PHASES = 2
NUM_LANES = 4

# Observation vector indices
PHASE_START, PHASE_END = 0, NUM_PHASES
MIN_GREEN_IDX = NUM_PHASES
DENSITY_START, DENSITY_END = MIN_GREEN_IDX + 1, MIN_GREEN_IDX + 1 + NUM_LANES
QUEUE_START, QUEUE_END = DENSITY_END, DENSITY_END + NUM_LANES

# Ensure SUMO_HOME is set
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

def get_neighboring_traffic_lights(net, ts_id):
    """
    Get neighboring traffic light IDs for a given traffic light.
    
    Args:
        net: The sumolib network object
        ts_id: The traffic light ID
    
    Returns:
        List of neighboring traffic light IDs
    """
    try:
        # Get the node corresponding to this traffic light
        node = net.getNode(ts_id)
        if node is None:
            return []
        
        # Get all incoming and outgoing edges from this node
        incoming_edges = node.getIncoming()
        outgoing_edges = node.getOutgoing()
        
        # Collect all connected edges
        connected_edges = list(incoming_edges) + list(outgoing_edges)
        
        # Find nodes at the other end of these edges and check if they have traffic lights
        neighboring_tls = set()
        
        for edge in connected_edges:
            # Get the nodes at both ends of the edge
            from_node = edge.getFromNode()
            to_node = edge.getToNode()
            
            # Check both nodes (excluding the current node)
            for other_node in [from_node, to_node]:
                if other_node and other_node.getID() != ts_id:
                    other_node_id = other_node.getID()
                    
                    # Check if this node has a traffic light
                    # We can check this by seeing if the node ID exists in our traffic light list
                    if other_node_id in [tls.getID() for tls in net.getTrafficLights()]:
                        neighboring_tls.add(other_node_id)
        
        return list(neighboring_tls)
        
    except Exception as e:
        print(f"Warning: Could not find neighbors for {ts_id}: {e}")
        return []

def compute_state(net, ts_id, obs, gru_stub_dim, max_neighbors):
    """
    Computes the state representation from the observation vector, including neighbor data,
    with padding for generalization.
    """
    local_obs = obs[ts_id]
    phase_one_hot = local_obs[PHASE_START:PHASE_END]
    local_queue = local_obs[QUEUE_START:QUEUE_END]
    
    state = np.concatenate([local_queue, phase_one_hot])

    # Get neighboring traffic lights using the corrected method
    neighbor_ids = get_neighboring_traffic_lights(net, ts_id)
    
    num_neighbors = 0
    for neighbor_id in neighbor_ids:
        if neighbor_id in obs:
            neighbor_obs = obs[neighbor_id]
            neighbor_phase = neighbor_obs[PHASE_START:PHASE_END]
            neighbor_queue = neighbor_obs[QUEUE_START:QUEUE_END]
            state = np.concatenate([state, neighbor_queue, neighbor_phase])
            num_neighbors += 1

    # Pad with zeros for remaining potential neighbors
    padding_size = (max_neighbors - num_neighbors) * (NUM_LANES + NUM_PHASES)
    if padding_size > 0:
        state = np.concatenate([state, np.zeros(padding_size)])

    predicted_inflow = np.random.rand(gru_stub_dim)
    state = np.concatenate([state, predicted_inflow])
    
    return state

def compute_reward(env, obs, prev_obs):
    """
    Computes a global reward based on the sum of queue lengths, virtual pedestrian waiting time, and fuel consumption.
    """
    reward = 0
    
    if REWARD_CONFIG["use_queue_length"]:
        total_queue = 0
        for ts_id in obs.keys():
            total_queue += np.sum(obs[ts_id][QUEUE_START:QUEUE_END])
        reward += total_queue * REWARD_CONFIG["queue_length_weight"]

    if REWARD_CONFIG["use_virtual_pedestrian_penalty"]:
        virtual_pedestrian_wait = 0
        for ts_id in obs.keys():
            virtual_pedestrian_wait += np.sum(obs[ts_id][QUEUE_START:QUEUE_END]) # Heuristic for pedestrian waiting
        reward += virtual_pedestrian_wait * REWARD_CONFIG["virtual_pedestrian_penalty"]

    if REWARD_CONFIG["use_fuel_consumption_penalty"]:
        total_fuel_consumption = 0
        for veh_id in env.sumo.vehicle.getIDList():
            total_fuel_consumption += env.sumo.vehicle.getFuelConsumption(veh_id)
        reward += total_fuel_consumption * REWARD_CONFIG["fuel_consumption_penalty"]

    if REWARD_CONFIG["use_jerk_penalty"]:
        for ts_id in obs.keys():
            if np.argmax(obs[ts_id][PHASE_START:PHASE_END]) != np.argmax(prev_obs[ts_id][PHASE_START:PHASE_END]):
                reward += REWARD_CONFIG["jerk_penalty"]

    return reward

def train():
    """Main training loop."""
    log_dir = os.path.join(TRAINING_CONFIG["log_dir"], datetime.now().strftime('%Y%m%d-%H%M%S'))
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(os.path.dirname(TRAINING_CONFIG["model_save_path"]), exist_ok=True)

    env = sumo_rl.SumoEnvironment(
        net_file='C:/Users/ultim/anaconda3/envs/metaworld-cpu/lib/site-packages/sumo_rl/nets/4x4-Lucas/4x4.net.xml',
        route_file='C:/Users/ultim/anaconda3/envs/metaworld-cpu/lib/site-packages/sumo_rl/nets/4x4-Lucas/4x4c1.rou.xml',
        out_csv_name=os.path.join(log_dir, "output.csv"),
        use_gui=SUMO_CONFIG["gui"],
        num_seconds=SUMO_CONFIG["num_seconds"],
        delta_time=SUMO_CONFIG["delta_time"],
        yellow_time=SUMO_CONFIG["yellow_time"],
        min_green=SUMO_CONFIG["min_green"],
        max_depart_delay=0
    )
    
    net = sumolib.net.readNet(env._net)

    initial_obs = env.reset()
    ts_ids = list(initial_obs.keys())
    
    # Calculate max_neighbors using the corrected method
    max_neighbors = 0
    for ts_id in ts_ids:
        neighbors = get_neighboring_traffic_lights(net, ts_id)
        if len(neighbors) > max_neighbors:
            max_neighbors = len(neighbors)

    first_ts_id = ts_ids[0]
    state_size = compute_state(net, first_ts_id, initial_obs, STUB_GRU_PREDICTION["inflow_dimension"], max_neighbors).shape[0]
    AGENT_CONFIG["state_size"] = state_size

    agent = D3QNAgent(state_size=AGENT_CONFIG["state_size"], action_size=AGENT_CONFIG["action_size"])
    epsilon = AGENT_CONFIG["epsilon_start"]
    best_reward = -np.inf

    for episode in range(1, TRAINING_CONFIG["episodes"] + 1):
        obs = env.reset()
        total_reward = 0
        done = {"__all__": False}
        prev_obs = obs
        
        while not done["__all__"]:
            action_dict = {}
            states = {}
            meta_actions = {}
            
            for ts_id in ts_ids:
                # Action masking
                if env.traffic_signals[ts_id].time_since_last_phase_change < env.traffic_signals[ts_id].min_green:
                    action_mask = np.array([1, 0]) # Force "hold"
                else:
                    action_mask = np.array([1, 1]) # Allow "hold" and "switch"

                states[ts_id] = compute_state(net, ts_id, obs, STUB_GRU_PREDICTION["inflow_dimension"], max_neighbors)
                meta_actions[ts_id] = agent.act(states[ts_id], epsilon, action_mask)
                
                current_phase = np.argmax(obs[ts_id][PHASE_START:PHASE_END])
                if meta_actions[ts_id] == 0: # Hold
                    action_dict[ts_id] = current_phase
                else: # Switch
                    action_dict[ts_id] = (current_phase + 1) % NUM_PHASES

            next_obs, _, done, _ = env.step(action_dict)

            reward = compute_reward(env, next_obs, prev_obs)
            
            for ts_id in ts_ids:
                next_state = compute_state(net, ts_id, next_obs, STUB_GRU_PREDICTION["inflow_dimension"], max_neighbors)
                agent.step(states[ts_id], meta_actions[ts_id], reward, next_state, done["__all__"])

            prev_obs = obs
            obs = next_obs
            total_reward += reward

        epsilon = max(AGENT_CONFIG["epsilon_end"], AGENT_CONFIG["epsilon_decay"] * epsilon)

        print(f"Episode {episode}/{TRAINING_CONFIG['episodes']} | Total Reward: {total_reward:.2f} | Epsilon: {epsilon:.4f}")

        if total_reward > best_reward:
            best_reward = total_reward
            agent.save(TRAINING_CONFIG["model_save_path"])
            print(f"New best model saved with reward: {best_reward:.2f}")

    env.close()

if __name__ == "__main__":
    train()