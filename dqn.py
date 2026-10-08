import random
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F





Transition = namedtuple("Transition", (
    "state", "action", "next_state", "next_action_mask", "reward", "discount"
))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = [None] * capacity
        self.size = 0
        self.position = 0

    def push(self, *args):
        """Insert a transition, replacing the oldest entry when full."""
        transition = Transition(*args)
        index = self.position
        if self.size < self.capacity:
            self.size += 1
        self.memory[index] = transition
        self.position = (index + 1) % self.capacity

    def sample(self, batch_size):
        """Uniformly sample distinct transitions from replay memory."""
        if batch_size > self.size:
            raise ValueError("Cannot sample more transitions than are stored.")
        # Sample indices so drawing a small batch does not copy the whole
        # replay buffer on every optimization step.
        indices = random.sample(range(self.size), batch_size)
        return [self.memory[index] for index in indices]

    def __len__(self):
        return self.size


class DQN(nn.Module):

    def __init__(self, n_observations, n_actions, hidden_size=16):
        super().__init__()
        self.n_actions = n_actions
        if hidden_size is None:
            # A linear head over a one-hot state index is an independent
            # state-action value table trained through the DQN update loop.
            self.layer1 = nn.Linear(n_observations, n_actions, bias=False)
            nn.init.zeros_(self.layer1.weight)
            self.layer2 = None
            self.layer3 = None
            return
        self.layer1 = nn.Linear(n_observations, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, n_actions)
        nn.init.zeros_(self.layer3.weight)
        nn.init.zeros_(self.layer3.bias)

    # Called with one state to choose an action or a batch during optimization.
    # Returns one Q estimate per action for each input state.
    def forward(self, x):
        if self.layer3 is None:
            return self.layer1(x)
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)
