
import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sumo_rl import SumoEnvironment
import traci

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

import argparse

class D3QN_Agent:
    def __init__(self, state_size, action_size, learning_rate=0.001, gamma=0.95, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = []
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

    def _build_model(self):
        # Dueling DQN
        input_layer = Input(shape=(self.state_size,))
        x = Dense(128, activation='relu')(input_layer)
        x = Dense(128, activation='relu')(x)

        # Value stream
        value_stream = Dense(1, activation='linear')(x)

        # Advantage stream
        advantage_stream = Dense(self.action_size, activation='linear')(x)
        advantage_stream = advantage_stream - tf.reduce_mean(advantage_stream, axis=1, keepdims=True)

        q_values = value_stream + advantage_stream
        model = Model(inputs=input_layer, outputs=q_values)
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        minibatch = np.random.choice(len(self.memory), batch_size, replace=False)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.target_model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, name):
        self.model.save_weights(name)

    def load(self, name):
        self.model.load_weights(name)

def custom_reward_function(ts_id):
    # Weights for each component of the reward function
    W_QUEUE = -0.5
    W_WAITING = -0.5
    W_FLOW = 1.0
    W_JERK = -0.1
    W_SWITCH = -0.5

    # Get controlled lanes
    lanes = traci.trafficlight.getControlledLanes(ts_id)
    
    # 1. Queue Length
    queue_length = 0
    for lane in lanes:
        queue_length += traci.lane.getLastStepHaltingNumber(lane)

    # 2. Waiting Time
    waiting_time = 0
    for lane in lanes:
        waiting_time += traci.lane.getWaitingTime(lane)

    # 3. Flow
    flow = 0
    for lane in lanes:
        flow += traci.lane.getLastStepVehicleNumber(lane)

    # 4. Jerk
    jerk = 0
    vehicles = traci.vehicle.getIDList()
    for vehicle_id in vehicles:
        if traci.vehicle.getLaneID(vehicle_id) in lanes:
            accel = traci.vehicle.getAcceleration(vehicle_id)
            # Approximate jerk as change in acceleration
            # This requires storing previous acceleration, for simplicity we use accel^2
            jerk += accel**2

    # 5. Frequent Switching
    # This needs to be handled in the environment or agent logic
    # For now, we'll add a placeholder. A simple way is to penalize any action that changes the phase.
    # A more sophisticated approach would be to check if the new phase is different from the old one.
    # This is handled by the environment's logic, so we can't directly implement it here.
    # We will assume the environment provides a penalty for switching.
    switch_penalty = 0 # Placeholder

    reward = (
        W_QUEUE * queue_length +
        W_WAITING * waiting_time +
        W_FLOW * flow +
        W_JERK * jerk +
        W_SWITCH * switch_penalty
    )

    return reward

class CommunicatingSumoEnvironment(SumoEnvironment):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.neighbor_radius = 1000  # 1 km
        self.neighbors = self._get_neighbors()

    def _get_neighbors(self):
        neighbors = {ts_id: [] for ts_id in self.ts_ids}
        positions = {ts_id: self.sumo.junction.getPosition(ts_id) for ts_id in self.ts_ids}

        for ts_id_1 in self.ts_ids:
            for ts_id_2 in self.ts_ids:
                if ts_id_1 != ts_id_2:
                    dist = np.linalg.norm(np.array(positions[ts_id_1]) - np.array(positions[ts_id_2]))
                    if dist < self.neighbor_radius:
                        neighbors[ts_id_1].append(ts_id_2)
        return neighbors

    def _compute_observations(self):
        observations = super()._compute_observations()
        communicating_observations = {}
        for ts_id in self.ts_ids:
            # Get local observation
            local_obs = observations[ts_id]

            # Get neighbor observations
            neighbor_obs = []
            for neighbor_id in self.neighbors[ts_id]:
                neighbor_obs.extend(observations[neighbor_id])
            
            # Concatenate local and neighbor observations
            communicating_observations[ts_id] = np.concatenate([local_obs] + neighbor_obs)
        return communicating_observations

import argparse
import json

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train D3QN agents for traffic signal control.')
    parser.add_argument('--resume', type=str, help='Path to the checkpoint file to resume training from.')
    args = parser.parse_args()

    env = CommunicatingSumoEnvironment(net_file='C:/Users/ultim/finalconfig/Bhubaneswar.net.xml',
                                     route_file='C:/Users/ultim/finalconfig/Bhubaneswar.rou.xml',
                                     out_csv_name='outputs/bhubaneswar_d3qn',
                                     use_gui=False,
                                     num_seconds=36000,  # 10 hours
                                     yellow_time=3,
                                     min_green=5,
                                     max_green=60,
                                     reward_fn=custom_reward_function)

    # Adjust state_size based on the number of neighbors
    base_state_size = env.observation_space.shape[0]
    ts_id_list = list(env.ts_ids)
    state_size = base_state_size * (1 + len(env.neighbors[ts_id_list[0]]))
    action_size = env.action_space.n

    # Create a D3QN agent for each traffic light
    agents = {ts_id: D3QN_Agent(state_size, action_size) for ts_id in env.ts_ids}

    start_episode = 0
    if args.resume:
        with open(args.resume, 'r') as f:
            checkpoint = json.load(f)
        start_episode = checkpoint['episode']
        for ts_id in env.ts_ids:
            agents[ts_id].load(checkpoint['models'][ts_id])
            agents[ts_id].epsilon = checkpoint['epsilons'][ts_id]

    best_reward = -np.inf

    # Training loop
    for e in range(start_episode, 100): # Number of episodes
        states = env.reset()
        done = {'__all__': False}
        total_reward = 0

        while not done['__all__']:
            actions = {ts_id: agents[ts_id].act(states[ts_id].reshape(1, -1)) for ts_id in env.ts_ids}
            next_states, rewards, done, _ = env.step(action=actions)

            for ts_id in env.ts_ids:
                agents[ts_id].remember(states[ts_id], actions[ts_id], rewards[ts_id], next_states[ts_id], done[ts_id])
                agents[ts_id].replay(32) # Batch size
                total_reward += rewards[ts_id]

            states = next_states

            if done['__all__']:
                print("Episode: {}/{}, Total Reward: {}, Epsilon: {:.2}".format(e, 100, total_reward, agents[env.ts_ids[0]].epsilon))
                
                # Save checkpoint
                checkpoint_data = {
                    'episode': e + 1,
                    'models': {ts_id: f'models/{ts_id}_d3qn_ep{e}.h5' for ts_id in env.ts_ids},
                    'epsilons': {ts_id: agents[ts_id].epsilon for ts_id in env.ts_ids}
                }
                for ts_id in env.ts_ids:
                    agents[ts_id].save(checkpoint_data['models'][ts_id])
                with open(f'checkpoints/d3qn_checkpoint_ep{e}.json', 'w') as f:
                    json.dump(checkpoint_data, f)

                # Save best model
                if total_reward > best_reward:
                    best_reward = total_reward
                    for ts_id in env.ts_ids:
                        agents[ts_id].save(f'models/{ts_id}_d3qn_best.h5')
                    print("New best model saved.")

                for ts_id in env.ts_ids:
                    agents[ts_id].update_target_model()

    env.close()
