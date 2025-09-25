
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from collections import namedtuple, deque

from config import AGENT_CONFIG

class QNetwork(nn.Module):
    """Dueling Q-Network for the Agent."""

    def __init__(self, state_size, action_size, fc1_units=64, fc2_units=64):
        """
        Initialize parameters and build model.
        Args:
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(QNetwork, self).__init__()
        self.action_size = action_size

        # Shared layers
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)

        # Dueling streams
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(fc2_units, fc2_units // 2),
            nn.ReLU(),
            nn.Linear(fc2_units // 2, 1)
        )

        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(fc2_units, fc2_units // 2),
            nn.ReLU(),
            nn.Linear(fc2_units // 2, self.action_size)
        )

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))

        value = self.value_stream(x)
        advantage = self.advantage_stream(x)

        # Combine value and advantage streams
        # Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size):
        """
        Initialize a ReplayBuffer object.
        Args:
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float()
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long()
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float()
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float()
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float()

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

class D3QNAgent:
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size):
        """
        Initialize an Agent object.
        Args:
            state_size (int): dimension of each state
            action_size (int): dimension of each action
        """
        self.state_size = state_size
        self.action_size = action_size
        self.device = AGENT_CONFIG['device']

        # Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size, AGENT_CONFIG['fc1_units'], AGENT_CONFIG['fc2_units']).to(self.device)
        self.qnetwork_target = QNetwork(state_size, action_size, AGENT_CONFIG['fc1_units'], AGENT_CONFIG['fc2_units']).to(self.device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=AGENT_CONFIG['lr'])

        # Replay memory
        self.memory = ReplayBuffer(AGENT_CONFIG['buffer_size'], AGENT_CONFIG['batch_size'])
        self.t_step = 0  # Initialize time step for updating network

    def step(self, state, action, reward, next_state, done):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        self.memory.add(state, action, reward, next_state, done)

        self.t_step = (self.t_step + 1) % AGENT_CONFIG['update_every']
        if self.t_step == 0:
            if len(self.memory) > AGENT_CONFIG['batch_size']:
                experiences = self.memory.sample()
                self.learn(experiences, AGENT_CONFIG['gamma'])

    def act(self, state, eps=0., action_mask=None):
        """
        Returns actions for given state as per current policy.
        Args:
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
            action_mask (array_like): mask to apply to the actions
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Apply action mask
        if action_mask is not None:
            action_values[action_mask == 0] = -1e8

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            # Choose from valid actions only
            valid_actions = np.where(action_mask == 1)[0] if action_mask is not None else np.arange(self.action_size)
            return random.choice(valid_actions)

    def learn(self, experiences, gamma):
        """
        Update value parameters using given batch of experience tuples.
        Args:
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences
        states, actions, rewards, next_states, dones = states.to(self.device), actions.to(self.device), rewards.to(self.device), next_states.to(self.device), dones.to(self.device)

        # Get best action from local network for next state
        q_local_next_actions = self.qnetwork_local(next_states).detach().max(1)[1].unsqueeze(1)
        
        # Get Q value for that action from target network
        q_targets_next = self.qnetwork_target(next_states).detach().gather(1, q_local_next_actions)

        # Compute Q targets for current states
        q_targets = rewards + (gamma * q_targets_next * (1 - dones))

        # Get expected Q values from local model
        q_expected = self.qnetwork_local(states).gather(1, actions)

        # Compute loss
        loss = F.mse_loss(q_expected, q_targets)
        
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, AGENT_CONFIG['tau'])

    def soft_update(self, local_model, target_model, tau):
        """
        Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Args:
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def save(self, path):
        """Save the model."""
        torch.save(self.qnetwork_local.state_dict(), path)
