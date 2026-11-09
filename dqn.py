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


class GRUDQN(nn.Module):
    """DQN head over a fixed window of observation feature vectors.

    Inputs have shape ``(batch, history_length, n_observations)``. The GRU
    summarizes each history and the linear head returns one Q value per action.
    """

    def __init__(self, n_observations, n_actions, hidden_size=64,
                 num_layers=1):
        super().__init__()
        if n_observations <= 0 or n_actions <= 0 or hidden_size <= 0:
            raise ValueError("Network dimensions must be positive")
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        self.n_observations = n_observations
        self.n_actions = n_actions
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(
            input_size=n_observations,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.q_head = nn.Linear(hidden_size, n_actions)
        nn.init.zeros_(self.q_head.weight)
        nn.init.zeros_(self.q_head.bias)

    def forward(self, x):
        if x.ndim == 2:
            # Treat a single (time, feature) sequence as a batch of one.
            x = x.unsqueeze(0)
        if x.ndim != 3:
            raise ValueError(
                "GRUDQN expects (batch, history, features) or (history, features)"
            )
        if x.shape[-1] != self.n_observations:
            raise ValueError(
                f"Expected {self.n_observations} observation features, "
                f"got {x.shape[-1]}"
            )
        _, hidden = self.gru(x)
        return self.q_head(hidden[-1])


class RSSMDQN(nn.Module):
    """DQN with a compact RSSM-inspired latent encoder over a fixed history.

    A fused GRU summarizes observation/action tokens. At each token, a
    categorical posterior combines recurrent context with the observation,
    while a prior predicts the categorical state from recurrent context alone.
    The DQN consumes the deterministic state and posterior probabilities.
    Small observation and reward heads provide auxiliary predictive losses.
    The categorical latent is not fed back into the deterministic transition;
    this is an encoder-focused approximation, not a full canonical RSSM.

    Histories have the same shape and semantics as :class:`GRUDQN`, allowing
    a matched fixed-window comparison with the existing replay buffer.
    """

    def __init__(self, n_observations, n_actions, observation_factors,
                 hidden_size=64, stochastic_variables=8,
                 categorical_classes=8):
        super().__init__()
        if min(n_observations, n_actions, hidden_size,
               stochastic_variables, categorical_classes) <= 0:
            raise ValueError("RSSM dimensions must be positive")
        self.n_observations = int(n_observations)
        self.n_actions = int(n_actions)
        self.hidden_size = int(hidden_size)
        self.stochastic_variables = int(stochastic_variables)
        self.categorical_classes = int(categorical_classes)
        self.stochastic_size = self.stochastic_variables * self.categorical_classes
        self.observation_factors = tuple(int(size) for size in observation_factors)
        self.encoder = nn.Sequential(
            nn.Linear(self.n_observations, self.hidden_size), nn.ELU()
        )
        self.transition = nn.GRU(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            batch_first=True,
        )
        self.prior = nn.Linear(self.hidden_size,
                               self.stochastic_variables * self.categorical_classes)
        self.posterior = nn.Linear(
            self.hidden_size + self.hidden_size,
            self.stochastic_variables * self.categorical_classes,
        )
        latent_size = self.hidden_size + self.stochastic_size
        self.q_head = nn.Linear(latent_size, self.n_actions)
        nn.init.zeros_(self.q_head.weight)
        nn.init.zeros_(self.q_head.bias)
        self.observation_head = nn.Linear(
            latent_size + self.n_actions, sum(self.observation_factors)
        )
        self.reward_head = nn.Linear(latent_size + self.n_actions, 1)

    def _encode(self, x):
        if x.ndim == 2:
            x = x.unsqueeze(0)
        if x.ndim != 3 or x.shape[-1] != self.n_observations:
            raise ValueError(
                "RSSMDQN expects (batch, history, features) with "
                f"{self.n_observations} features"
            )
        _batch, _history, _ = x.shape
        encoded = self.encoder(x)
        deterministic, _hidden = self.transition(encoded)
        prior_logits = self.prior(deterministic).view(
            x.shape[0], x.shape[1], self.stochastic_variables,
            self.categorical_classes,
        )
        posterior_logits = self.posterior(
            torch.cat((deterministic, encoded), dim=-1)
        ).view(
            x.shape[0], x.shape[1], self.stochastic_variables,
            self.categorical_classes,
        )
        prior_prob = prior_logits.softmax(dim=-1)
        posterior_prob = posterior_logits.softmax(dim=-1)
        stochastic = posterior_prob.flatten(start_dim=2)
        latent_sequence = torch.cat((deterministic, stochastic), dim=-1)
        valid_index = sum(self.observation_factors)
        kl_values = (posterior_prob * (
            posterior_prob.clamp_min(1e-8).log()
            - prior_prob.clamp_min(1e-8).log()
        )).sum(dim=-1).sum(dim=-1)
        valid = x[:, :, valid_index]
        kl_mean = (kl_values * valid).sum() / valid.sum().clamp_min(1.0)
        return latent_sequence[:, -1], kl_mean

    def forward(self, x):
        q_values, _latent, _kl = self.forward_features(x)
        return q_values

    def forward_features(self, x):
        """Return Q values and shared latent/KL features for one model pass."""
        latent, kl = self._encode(x)
        return self.q_head(latent), latent, kl

    def auxiliary_loss(self, history, actions, next_history, rewards, *,
                       observation_weight=0.1, reward_weight=0.1,
                       kl_weight=0.01, observation_indices=None):
        """Return auxiliary loss and detached diagnostics for a replay batch."""
        latent, kl = self._encode(history)
        return self.auxiliary_loss_from_features(
            latent, kl, actions, next_history, rewards,
            observation_weight=observation_weight,
            reward_weight=reward_weight,
            kl_weight=kl_weight,
            observation_indices=observation_indices,
        )

    def auxiliary_loss_from_features(
            self, latent, kl, actions, next_history, rewards, *,
            observation_weight=0.1, reward_weight=0.1, kl_weight=0.01,
            observation_indices=None):
        """Compute prediction losses from a previously encoded replay batch."""
        action_one_hot = F.one_hot(
            actions.reshape(-1), num_classes=self.n_actions
        ).to(dtype=latent.dtype)
        prediction_input = torch.cat((latent, action_one_hot), dim=-1)
        reward_target = (rewards.reshape(-1) > 0).to(dtype=latent.dtype)
        reward_logits = self.reward_head(prediction_input).squeeze(-1)
        reward_loss = F.binary_cross_entropy_with_logits(
            reward_logits, reward_target
        )

        if next_history is not None and next_history.shape[0] > 0:
            next_token = next_history[:, -1]
            observation_rows = (
                torch.arange(next_history.shape[0], device=latent.device)
                if observation_indices is None else observation_indices
            )
            prediction = self.observation_head(prediction_input[observation_rows])
            observation_loss = latent.new_zeros(())
            offset = 0
            for size in self.observation_factors:
                target = next_token[:, offset:offset + size].argmax(dim=-1)
                observation_loss = observation_loss + F.cross_entropy(
                    prediction[:, offset:offset + size], target
                )
                offset += size
        else:
            observation_loss = latent.new_zeros(())
        total = (
            observation_weight * observation_loss
            + reward_weight * reward_loss
            + kl_weight * kl
        )
        diagnostics = {
            "observation_loss": float(observation_loss.detach()),
            "reward_loss": float(reward_loss.detach()),
            "kl": float(kl.detach()),
        }
        return total, diagnostics
