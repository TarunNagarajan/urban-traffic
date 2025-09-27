
import torch
import torch.nn as nn
from torch.distributions import Categorical
import torch.optim as optim

class ActorCritic(nn.Module):
    """PPO Actor-Critic Network."""
    def __init__(self, state_size, action_size, fc1_units=128, fc2_units=128):
        super(ActorCritic, self).__init__()
        self.shared_layers = nn.Sequential(
            nn.Linear(state_size, fc1_units),
            nn.ReLU(),
            nn.Linear(fc1_units, fc2_units),
            nn.ReLU()
        )
        self.actor_head = nn.Linear(fc2_units, action_size)
        self.critic_head = nn.Linear(fc2_units, 1)

    def forward(self, state):
        x = self.shared_layers(state)
        action_logits = self.actor_head(x)
        state_value = self.critic_head(x)
        return action_logits, state_value

class PPOAgent:
    """PPO Agent."""
    def __init__(self, state_size, action_size, lr, gamma, ppo_clip, num_epochs, batch_size, device):
        self.device = device
        self.gamma = gamma
        self.ppo_clip = ppo_clip
        self.num_epochs = num_epochs
        self.batch_size = batch_size

        self.policy = ActorCritic(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.mse_loss = nn.MSELoss()

    def get_action_and_value(self, state, action=None):
        action_logits, state_value = self.policy(state)
        dist = Categorical(logits=action_logits)
        
        if action is None:
            action = dist.sample()
            
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        
        return action, log_prob, entropy, state_value

    def update(self, states, actions, old_log_probs, returns, advantages):
        """Update the policy using a batch of trajectories."""
        for _ in range(self.num_epochs):
            # Create minibatches
            for i in range(0, len(states), self.batch_size):
                end = i + self.batch_size
                mb_states = states[i:end]
                mb_actions = actions[i:end]
                mb_old_log_probs = old_log_probs[i:end]
                mb_returns = returns[i:end]
                mb_advantages = advantages[i:end]

                # Get new log_probs, entropy, and values
                _, new_log_probs, entropy, new_values = self.get_action_and_value(mb_states, mb_actions)

                # --- Calculate Actor (Policy) Loss ---
                ratio = (new_log_probs - mb_old_log_probs).exp()
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.ppo_clip, 1.0 + self.ppo_clip) * mb_advantages
                actor_loss = -torch.min(surr1, surr2).mean()

                # --- Calculate Critic (Value) Loss ---
                critic_loss = self.mse_loss(new_values, mb_returns)

                # --- Calculate Total Loss ---
                # The entropy bonus encourages exploration
                entropy_bonus = entropy.mean()
                loss = critic_loss * 0.5 + actor_loss - entropy_bonus * 0.01

                # --- Update ---
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
    
    def save(self, path):
        torch.save(self.policy.state_dict(), path)

    def load(self, path):
        self.policy.load_state_dict(torch.load(path, map_location=self.device))
