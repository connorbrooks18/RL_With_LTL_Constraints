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

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super().__init__()
        self.n_actions = n_actions
        self.layer1 = nn.Linear(n_observations, 16)
        self.layer2 = nn.Linear(16, 16)
        self.layer3 = nn.Linear(16, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        
        x = F.relu(self.layer2(x))
        return self.layer3(x)
