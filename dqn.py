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
        return random.sample(self.memory[:self.size], batch_size)

    def __len__(self):
        return self.size


class DQN(nn.Module):

    def __init__(self, n_observations, n_actions, hidden_size=16):
        super().__init__()
        self.n_actions = n_actions
        self.layer1 = nn.Linear(n_observations, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, n_actions)
        # Match the tabular learner's zero initialization without constraining
        # the network's predictions. A linear head also leaves overshoot visible.
        nn.init.zeros_(self.layer3.weight)
        nn.init.zeros_(self.layer3.bias)

    # Called with one state to choose an action or a batch during optimization.
    # Returns one Q estimate per action for each input state.
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        # Keep a linear Q head: a sigmoid would cap values but also hide the
        # amount of overestimation and shrink gradients near 0 and 1.
        return self.layer3(x)
