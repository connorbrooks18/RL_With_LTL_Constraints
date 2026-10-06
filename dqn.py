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
        # Store slot indices so sampling does not scan the full replay buffer.
        self.reward_memory = _IndexPool()
        self.epsilon_memory = _IndexPool()

    def push(self, *args, reward_event=False, epsilon_action=False):
        """Insert a transition, replacing the oldest entry when full."""
        transition = Transition(*args)
        index = self.position
        if self.size == self.capacity:
            self.reward_memory.discard(index)
            self.epsilon_memory.discard(index)
        else:
            self.size += 1
        self.memory[index] = transition
        if reward_event:
            self.reward_memory.add(index)
        if epsilon_action:
            self.epsilon_memory.add(index)
        self.position = (index + 1) % self.capacity

    def sample(self, batch_size, reward_fraction=0.25, epsilon_fraction=0.25):
        """Sample with event emphasis without scanning the whole buffer."""
        if batch_size > self.size:
            raise ValueError("Cannot sample more transitions than are stored.")

        selected = []
        selected_indices = set()
        n_reward = min(int(batch_size * reward_fraction), len(self.reward_memory))
        reward_indices = self.reward_memory.sample_excluding(n_reward, selected_indices)
        selected.extend(reward_indices)
        selected_indices.update(reward_indices)

        n_epsilon = min(int(batch_size * epsilon_fraction),
                        len(self.epsilon_memory) - self.epsilon_memory.count_in(selected_indices))
        epsilon_indices = self.epsilon_memory.sample_excluding(n_epsilon, selected_indices)
        selected.extend(epsilon_indices)
        selected_indices.update(epsilon_indices)

        remaining = batch_size - len(selected)
        if remaining:
            selected.extend(_sample_range_excluding(self.size, remaining, selected_indices))
        return [self.memory[index] for index in selected]

    def __len__(self):
        return self.size


class _IndexPool:
    """A set of replay slots with O(1) insertion and removal."""

    def __init__(self):
        self.indices = []
        self.positions = {}

    def __len__(self):
        return len(self.indices)

    def add(self, index):
        if index not in self.positions:
            self.positions[index] = len(self.indices)
            self.indices.append(index)

    def discard(self, index):
        position = self.positions.pop(index, None)
        if position is None:
            return
        last = self.indices.pop()
        if position < len(self.indices):
            self.indices[position] = last
            self.positions[last] = position

    def count_in(self, indices):
        return sum(index in self.positions for index in indices)

    def sample_excluding(self, count, excluded):
        available = len(self.indices) - self.count_in(excluded)
        count = min(count, available)
        if count <= 0:
            return []
        if count == available:
            candidates = [index for index in self.indices if index not in excluded]
            random.shuffle(candidates)
            return candidates

        result = []
        blocked = set(excluded)
        attempts = 0
        attempt_limit = max(64, count * 8)
        while len(result) < count and attempts < attempt_limit:
            index = random.choice(self.indices)
            if index not in blocked:
                result.append(index)
                blocked.add(index)
            attempts += 1
        if len(result) < count:
            candidates = [index for index in self.indices if index not in blocked]
            result.extend(random.sample(candidates, count - len(result)))
        return result


def _sample_range_excluding(population_size, count, excluded):
    """Draw unique replay slots uniformly, usually in O(count) time."""
    available = population_size - len(excluded)
    count = min(count, available)
    if count <= 0:
        return []
    result = []
    blocked = set(excluded)
    attempts = 0
    attempt_limit = max(64, count * 8)
    while len(result) < count and attempts < attempt_limit:
        index = random.randrange(population_size)
        if index not in blocked:
            result.append(index)
            blocked.add(index)
        attempts += 1
    if len(result) < count:
        candidates = [index for index in range(population_size) if index not in blocked]
        result.extend(random.sample(candidates, count - len(result)))
    return result


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
