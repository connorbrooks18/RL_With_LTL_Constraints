import random
from collections import namedtuple, deque

import torch
import torch.nn as nn
import torch.nn.functional as F





Transition = namedtuple("Transition", (
    "state", "action", "next_state", "next_action_mask", "reward", "discount"
))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)
        # Keep separate, bounded pools for sparse but informative events.
        # These hold references to transitions also stored in `memory`.
        self.reward_memory = deque([], maxlen=capacity)
        self.epsilon_memory = deque([], maxlen=capacity)

    def push(self, *args, reward_event=False, epsilon_action=False):
        """Save a transition"""
        transition = Transition(*args)
        self.memory.append(transition)
        if reward_event:
            self.reward_memory.append(transition)
        if epsilon_action:
            self.epsilon_memory.append(transition)

    def sample(self, batch_size, reward_fraction=0.25, epsilon_fraction=0.25):
        """Sample a uniform batch with capped reward/epsilon-event emphasis."""
        # Specialized pools may retain entries that have since fallen out of
        # the rolling replay buffer. Restrict every batch to current entries.
        current_ids = {id(transition) for transition in self.memory}
        reward_candidates = [t for t in self.reward_memory if id(t) in current_ids]
        epsilon_candidates = [t for t in self.epsilon_memory if id(t) in current_ids]

        batch = []
        n_reward = min(int(batch_size * reward_fraction), len(reward_candidates))
        if n_reward:
            batch.extend(random.sample(reward_candidates, n_reward))

        selected_ids = {id(transition) for transition in batch}
        remaining_epsilon = [t for t in epsilon_candidates if id(t) not in selected_ids]
        n_epsilon = min(int(batch_size * epsilon_fraction), len(remaining_epsilon))
        if n_epsilon:
            batch.extend(random.sample(remaining_epsilon, n_epsilon))

        selected_ids = {id(transition) for transition in batch}
        remaining = [t for t in self.memory if id(t) not in selected_ids]
        batch.extend(random.sample(remaining, batch_size - len(batch)))
        return batch

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):

    def __init__(self, n_observations, n_actions, hidden_size=128):
        super().__init__()
        self.n_actions = n_actions
        self.layer1 = nn.Linear(n_observations, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, n_actions)

    # Called with one state to choose an action or a batch during optimization.
    # Returns one bounded Q estimate per action for each input state.
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        # The accepting reward is 1-gammaB and uses gammaB as its discount,
        # so the exact return is in [0, 1]. Keep function approximation inside
        # the same range as the tabular Bellman solution.
        return torch.sigmoid(self.layer3(x))
