"""Control Synthesis using Reinforcement Learning.
"""
import numpy as np
from itertools import product
from mdp import GridMDP
import os
import importlib
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import namedtuple, deque
import random

if importlib.util.find_spec('matplotlib'):
    import matplotlib.pyplot as plt
    
if importlib.util.find_spec('ipywidgets'):
    from ipywidgets.widgets import IntSlider
    from ipywidgets import interact



class ControlSynthesis:
    """This class is the implementation of our main control synthesis algorithm.
    
    Attributes
    ----------
    shape : (n_pairs, n_qs, n_rows, n_cols, n_actions)
        The shape of the product MDP.
    
    reward : array, shape=(n_pairs,n_qs,n_rows,n_cols)
        The reward function. Accepting states receive 1-discount_accepting; all other states receive 0.
        
    transition_probs : array, shape=(n_pairs,n_qs,n_rows,n_cols,n_actions)
        The transition probabilities. self.transition_probs[state][action] stores a pair of lists ([s1,s2,..],[p1,p2,...]) that contains only positive probabilities and the corresponding transitions.
    
    Parameters
    ----------
    mdp : mdp.GridMDP
        The MDP that models the environment.
        
    oa : oa.OmegaAutomatan
        The OA obtained from the LTL specification.
        
    discount_nonaccepting : float
        Discount used outside the accepting set (paper's gamma).
    
    discount_accepting : float
        Discount used in accepting states (paper's gamma_B). The suffix refers
        to the paper's accepting set B, not GridMDP's blocked-cell label 'B'.
    
    """
    def __init__(self, mdp, oa, discount_nonaccepting=0.99999,
                 discount_accepting=0.99, *, discount=None, discountB=None):
        # Keep the original keyword names working for existing notebooks.
        if discount is not None:
            discount_nonaccepting = discount
        if discountB is not None:
            discount_accepting = discountB
        self.mdp = mdp
        self.oa = oa
        self.discount_nonaccepting = discount_nonaccepting
        self.discount_accepting = discount_accepting
        # Backward-compatible attribute aliases.
        self.discount = self.discount_nonaccepting
        self.discountB = self.discount_accepting
        self.shape = oa.shape + mdp.shape + (len(mdp.A)+oa.shape[1],)
        
        # Create the action matrix
        self.A = np.empty(self.shape[:-1],dtype=object)
        for i,q,r,c in self.states():
            self.A[i,q,r,c] = list(range(len(mdp.A))) + [len(mdp.A)+e_a for e_a in oa.eps[q]]
        
        # Create the reward matrix
        self.reward = np.zeros(self.shape[:-1])
        for i,q,r,c in self.states():
            self.reward[i,q,r,c] = (1-self.discount_accepting
                                    if oa.acc[q][mdp.label[r,c]][i] else 0)
        
        # Create the transition matrix
        self.transition_probs = np.empty(self.shape,dtype=object)  # Enrich the action set with epsilon-actions
        for i,q,r,c in self.states():
            for action in self.A[i,q,r,c]:
                if action < len(self.mdp.A): # MDP actions
                    q_ = oa.delta[q][mdp.label[r,c]]  # OA transition
                    mdp_states, probs = mdp.get_transition_prob((r,c),mdp.A[action])  # MDP transition
                    self.transition_probs[i,q,r,c][action] = [(i,q_,)+s for s in mdp_states], probs  
                else:  # epsilon-actions
                    self.transition_probs[i,q,r,c][action] = ([(i,action-len(mdp.A),r,c)], [1.])
                    # epsilon here is only from q0->q1?
    
    def states(self):
        """State generator.
        
        Yields
        ------
        state: tuple
            State coordinates (i,q,r,c)).
        """
        n_mdps, n_qs, n_rows, n_cols, n_actions = self.shape
        for i,q,r,c in product(range(n_mdps),range(n_qs),range(n_rows),range(n_cols)):
            yield i,q,r,c
    
    def random_state(self):
        """Generates a random state coordinate.
        
        Returns
        -------
        state: tuple
            A random state coordinate (i,q,r,c).
        """
        n_mdps, n_qs, n_rows, n_cols, n_actions = self.shape
        mdp_state = np.random.randint(n_rows),np.random.randint(n_cols)
        return (np.random.randint(n_mdps),np.random.randint(n_qs)) + mdp_state

    def start_states(self):
        """Return every grid cell as an eligible episode start.

        This matches GridMDP.random_state: all cells are sampled uniformly,
        including proposition-labeled and blocked cells.
        """
        return list(self.mdp.states())

    def observe_state(self, state, error_prob=None):
        """Return a noisy product-state observation, retaining automaton q.

        This is the legacy fully-observed product-state interface. Use
        :meth:`observe_observation` for a POMDP observation, which does not
        reveal the hidden automaton state.
        """
        i, q, r, c = state
        observed_r, observed_c = self.mdp.observe_state((r, c), error_prob)
        return (i, q, observed_r, observed_c)

    def observe_observation(self, state, error_prob=None):
        """Return the visible observation ``(pair, row, column)``.

        The true automaton state is intentionally omitted. Rewards and
        product-state transitions continue to use the true state.
        """
        i, _q, r, c = state
        observed_r, observed_c = self.mdp.observe_state((r, c), error_prob)
        return (i, observed_r, observed_c)

    def encode_observation(self, observation, *, partial_observation=True,
                           previous_action=None, recurrent=False):
        """One-hot encode an observation for a DQN or recurrent DQN.

        For partial observations, pass ``(pair, row, column)``. For full
        product-state observations, pass ``(pair, q, row, column)``. Recurrent
        tokens append a valid-token bit and, when present, the preceding action
        as a one-hot vector. Left-padding tokens are all zeros; the history
        encoder appends persistent one-hot epsilon-target memory separately.
        """
        observation = tuple(map(int, observation))
        if partial_observation:
            if len(observation) != 3:
                raise ValueError("Partial observations must be (pair, row, column)")
            factors = (self.shape[0], self.mdp.shape[0], self.mdp.shape[1])
        else:
            if len(observation) != 4:
                raise ValueError("Product observations must be (pair, q, row, column)")
            factors = self.shape[:-1]

        feature_count = sum(factors) + (1 + self.shape[-1] if recurrent else 0)
        encoded = torch.zeros(feature_count, dtype=torch.float32)
        offset = 0
        for coordinate, size in zip(observation, factors):
            if coordinate < 0 or coordinate >= size:
                raise ValueError(f"Observation coordinate {coordinate} is out of range")
            encoded[offset + coordinate] = 1.0
            offset += size

        if recurrent:
            encoded[offset] = 1.0  # distinguishes a real all-zero-like token from padding
            if previous_action is not None:
                previous_action = int(previous_action)
                if previous_action < 0 or previous_action >= self.shape[-1]:
                    raise ValueError("previous_action is outside the DQN action range")
                encoded[offset + 1 + previous_action] = 1.0
        return encoded

    def encode_observation_history(self, observations, history_length, *,
                                   previous_actions=None,
                                   partial_observation=True,
                                   epsilon_committed=False,
                                   epsilon_target=None,
                                   epsilon_commitment_mode=None,
                                   include_epsilon_commitment=None):
        """Encode and left-pad a history for :class:`dqn.GRUDQN`.

        ``observations`` are ordered oldest to newest. ``previous_actions``
        contains the action taken immediately before each corresponding
        observation, or ``None`` for the first observation after reset.
        Histories longer than ``history_length`` are truncated from the left.
        By default, persistent one-hot features record the target of the most
        recent epsilon edge chosen by the controller. This is derived from
        its own actions, not from the hidden automaton state. The legacy
        ``bit`` mode supports checkpoints that stored only whether an epsilon
        edge had been chosen; ``none`` supports older checkpoints.
        """
        if history_length <= 0:
            raise ValueError("history_length must be positive")
        factors = (
            (self.shape[0], self.mdp.shape[0], self.mdp.shape[1])
            if partial_observation else self.shape[:-1]
        )
        observations = list(observations)
        if previous_actions is None:
            previous_actions = [None] * len(observations)
        else:
            previous_actions = list(previous_actions)
            if len(previous_actions) != len(observations):
                raise ValueError("observations and previous_actions must have equal length")
        if not observations:
            raise ValueError("At least one observation is required")

        if epsilon_commitment_mode is None:
            if include_epsilon_commitment is False:
                epsilon_commitment_mode = "none"
            elif include_epsilon_commitment is True:
                # Preserve the old public option for existing notebooks.
                epsilon_commitment_mode = "bit"
            else:
                epsilon_commitment_mode = "target_one_hot"
        if epsilon_commitment_mode not in ("target_one_hot", "bit", "none"):
            raise ValueError(
                "epsilon_commitment_mode must be 'target_one_hot', 'bit', or 'none'"
            )
        if epsilon_target is None:
            epsilon_actions = [
                int(action) for action in previous_actions
                if action is not None and int(action) >= len(self.mdp.A)
            ]
            if epsilon_actions:
                epsilon_target = epsilon_actions[-1] - len(self.mdp.A)
        if epsilon_target is not None:
            epsilon_target = int(epsilon_target)
            if epsilon_target < 0 or epsilon_target >= self.shape[1]:
                raise ValueError("epsilon_target is outside the automaton state range")
        if (epsilon_commitment_mode == "target_one_hot"
                and epsilon_committed and epsilon_target is None):
            raise ValueError(
                "target_one_hot commitment needs epsilon_target or an epsilon action "
                "in previous_actions"
            )

        tokens = [
            self.encode_observation(
                observation,
                partial_observation=partial_observation,
                previous_action=action,
                recurrent=True,
            )
            for observation, action in zip(observations, previous_actions)
        ][-history_length:]
        if len(tokens) < history_length:
            padding = [torch.zeros_like(tokens[0])
                       for _ in range(history_length - len(tokens))]
            tokens = padding + tokens
        encoded_history = torch.stack(tokens)
        if epsilon_commitment_mode == "target_one_hot":
            valid_token_index = sum(factors)
            valid_tokens = encoded_history[:, valid_token_index:valid_token_index + 1]
            target_features = torch.zeros(
                self.shape[1], dtype=torch.float32
            )
            if epsilon_target is not None:
                target_features[epsilon_target] = 1.0
            commitment = valid_tokens * target_features.unsqueeze(0)
            encoded_history = torch.cat((encoded_history, commitment), dim=1)
        elif epsilon_commitment_mode == "bit":
            valid_token_index = sum(factors)
            valid_tokens = encoded_history[:, valid_token_index:valid_token_index + 1]
            commitment = valid_tokens * float(bool(epsilon_committed or epsilon_target is not None))
            encoded_history = torch.cat((encoded_history, commitment), dim=1)
        return encoded_history
    
    def q_learning(self,start=None,T=None,K=None):
        """Performs the Q-learning algorithm and returns the action values.
        
        Parameters
        ----------
        start : int
            The start state of the MDP.
            
        T : int
            The episode length.
        
        K : int 
            The number of episodes.
            
        Returns
        -------
        Q: array, shape=(n_pairs,n_qs,n_rows,n_cols,n_actions) 
            The action values learned.
        """
        
        # T ends a rollout and resets to its configured start; it does not
        # make the last transition terminal for the Bellman update.
        T = T if T else np.prod(self.shape[:-1])
        K = K if K else 100000

        start_states = self.start_states()
        if not start_states:
            raise ValueError("Cannot train: the grid has no start states.")

        Q = np.zeros(self.shape)

        for k in range(K):
            if((k%2500) == 0): print(k)
            mdp_start = (start if start is not None else
                         start_states[np.random.randint(len(start_states))])
            state = (self.shape[0]-1,self.oa.q0)+mdp_start
            progress = k / max(1, K - 1)
            alpha = max(1.0 - 0.999 * progress, 0.001)
            epsilon = max(1.0 - 0.9 * progress, 0.1)
            for t in range(T):

                reward = self.reward[state]
                gamma = self.discount_accepting if reward else self.discount_nonaccepting
                
                # Follow an epsilon-greedy policy
                legal_actions = self.A[state]
                if np.random.rand() < epsilon or max(Q[state][a] for a in legal_actions) == 0:
                    action = np.random.choice(legal_actions)  # Choose among the MDP and epsilon actions
                else:
                    action = legal_actions[np.argmax([Q[state][a] for a in legal_actions])]
                
                # Observe the next state
                states, probs = self.transition_probs[state][action]
                next_state = states[np.random.choice(len(states),p=probs)]
                
                # Q-update
                next_legal_actions = self.A[next_state]
                next_value = max(Q[next_state][a] for a in next_legal_actions)
                Q[state][action] += alpha * (reward + gamma*next_value - Q[state][action])

                state = next_state
        
        return Q



        
    
    def greedy_policy(self,value):
        """Returns a greedy policy for the given value function.
        
        Parameters
        ----------
        value: array, size=(n_pairs,n_qs,n_rows,n_cols)
            The value function.
        
        Returns
        -------
        policy : array, size=(n_pairs,n_qs,n_rows,n_cols)
            The policy.
        
        """
        policy = np.zeros((value.shape),dtype=int)
        for state in self.states():
            action_values = np.empty(len(self.A[state]))
            for i,action in enumerate(self.A[state]):
                action_values[i] = np.sum([value[s]*p for s,p in zip(*self.transition_probs[state][action])])
            policy[state] = self.A[state][np.argmax(action_values)]
        return policy
    
    def value_iteration(self,T=None,threshold=None):
        """Performs the value iteration algorithm and returns the value function. It requires at least one parameter.
        
        Parameters
        ----------
        T : int
            The number of iterations.
        
        threshold: float
            The threshold value to be used in the stopping condition.
        
        Returns
        -------
        value: array, size=(n_mdps,n_qs,n_rows,n_cols)
            The value function.
        """
        states = list(self.states())
        state_indices = {state: index for index, state in enumerate(states)}
        n_states = len(states)
        n_actions = self.shape[-1]
        n_outcomes = max(
            len(self.transition_probs[state][action][0])
            for state in states for action in self.A[state]
        )

        # Pad legal action outcomes into dense arrays once. The Bellman sweep
        # can then run over all states and actions with NumPy instead of
        # rebuilding Python lists for every state on every iteration.
        next_indices = np.zeros((n_states, n_actions, n_outcomes), dtype=np.intp)
        probabilities = np.zeros((n_states, n_actions, n_outcomes), dtype=float)
        legal_actions = np.zeros((n_states, n_actions), dtype=bool)
        rewards = np.empty(n_states, dtype=float)
        discounts = np.empty(n_states, dtype=float)
        for index, state in enumerate(states):
            rewards[index] = self.reward[state]
            discounts[index] = (self.discount_accepting if rewards[index] > 0
                                else self.discount_nonaccepting)
            for action in self.A[state]:
                legal_actions[index, action] = True
                successors, probs = self.transition_probs[state][action]
                count = len(successors)
                next_indices[index, action, :count] = [state_indices[s] for s in successors]
                probabilities[index, action, :count] = probs

        old_value = np.zeros(n_states, dtype=float)
        t = 0
        d = np.inf
        while (T and t < T) or (threshold and d > threshold):
            successor_values = old_value[next_indices]
            action_values = np.sum(successor_values * probabilities, axis=2)
            action_values[~legal_actions] = -np.inf
            value = rewards + discounts * np.max(action_values, axis=1)
            t += 1
            d = np.max(np.abs(old_value - value))
            old_value = value

        return old_value.reshape(self.shape[:-1])
    
    def simulate(self,value,policy,start=None,T=None,plot=True, animation=None):
        """Simulates the environment and returns a trajectory obtained under the given policy.
        
        Parameters
        ----------
        policy : array, size=(n_pairs,n_qs,n_rows,n_cols)
            The policy.
        
        start : int
            The start state of the MDP.
            
        T : int
            The episode length.
        
        plot : bool 
            Plots the simulation if it is True.
            
        Returns
        -------
        episode: list
            A sequence of states
        """
        T = T if T else np.prod(self.shape[:-1])
        state = (self.shape[0]-1,self.oa.q0)+(start if start else self.mdp.random_state())
        episode = [state]
        for t in range(T):
            states, probs = self.transition_probs[state][policy[state]]
            state = states[np.random.choice(len(states),p=probs)]
            episode.append(state)
            
        if plot:
            def plot_agent(t):
                self.mdp.plot(value=value[episode[t][:2]],
                              policy=policy[episode[t][:2]],
                              agent=episode[t][2:])
            t=IntSlider(value=0,min=0,max=T-1)
            interact(plot_agent,t=t)
            
        if animation: # see cotrolsynanimation.py
            for t in range(T):
                filenumber = format(t,"05")
                self.mdp.plot(value=value[episode[t][:2]],
                              policy=policy[episode[t][:2]],
                              agent=episode[t][2:],save="./animation/image{}.png".format(filenumber))
                plt.close()
            os.system("ffmpeg -f image2 -r 5 -i ./animation/image%05d.png -vcodec mpeg4 -y safeAbsorbingStates.avi")
        
        return episode
        
    def plot(self, value=None, policy=None, iq=None, **kwargs):
        """Plots the values of the states as a color matrix with two sliders.
        
        Parameters
        ----------
        value : array, shape=(n_mdps,n_qs,n_rows,n_cols) 
            The value function.
            
        policy : array, shape=(n_mdps,n_qs,n_rows,n_cols) 
            The policy to be visualized. It is optional.
            
        save : str
            The name of the file the image will be saved to. It is optional
        """
        
        if iq:
            val = value[iq] if value is not None else None
            pol = policy[iq] if policy is not None else None
            self.mdp.plot(val,pol,**kwargs)
        else:
            # A helper function for the sliders
            def plot_value(i,q):
                val = value[i,q] if value is not None else None
                pol = policy[i,q] if policy is not None else None
                self.mdp.plot(val,pol,**kwargs)
            i = IntSlider(value=0,min=0,max=self.shape[0]-1)
            q = IntSlider(value=self.oa.q0,min=0,max=self.shape[1]-1)
            interact(plot_value,i=i,q=q)

    def deep_q_learning(self, start=None, T=None, K=None, *,
                        epsilon_floor=0.1, learning_rate=1e-4,
                        automaton_action_probability=0.2,
                        return_diagnostics=False, architecture="linear",
                        replay_capacity=100_000, batch_size=128,
                        target_update_every=250,
                        checkpoint_dir="checkpoints", history_length=5,
                        gru_hidden_size=64, partial_observation=None,
                        return_model=False):
        """Train a DQN, optionally from partial observations and fixed histories.

        ``architecture="gru"`` consumes the most recent ``history_length``
        observation tokens. In partial-observation mode each token omits the
        true automaton state and includes the preceding action, so the network
        can track recent automaton progress caused by its own epsilon choices.
        The hidden state is recomputed from this fixed window at each decision;
        information older than ``history_length`` is not retained.
        When epsilon commitment determines future epsilon availability from
        the controller's own actions, the mask follows that observable history.
        Otherwise unavailable epsilon attempts are no-ops under a global mask.
        A proven zero-value rejecting sink stops bootstrapping in the replay
        target only; the sink state is never added to the policy input.

        If ``partial_observation`` is omitted, it is enabled for GRU runs and
        whenever the MDP has nonzero observation noise. For legacy exact-state
        linear and MLP runs it defaults to the full product state.
        """
        import random
        import dqn
        from dqn import Transition
        # Actions are selected one state at a time, where MPS launch and
        # synchronization costs dominate this small network. Use CPU for this
        # sequential loop; it benchmarks substantially faster on the thesis
        # grid while batched replay updates remain inexpensive.
        device = torch.device("cpu")
        # T is a rollout/reset horizon, not a terminal condition. Grid traps
        # continue to accrue accepting rewards and must not be marked done.
        T = T if T else int(np.prod(self.shape[:-1]))
        K = K if K else 100000
        update_every = 4
        n_actions = self.shape[-1]
        if architecture not in ("linear", "mlp", "gru"):
            raise ValueError("architecture must be 'linear', 'mlp', or 'gru'")
        if history_length <= 0:
            raise ValueError("history_length must be positive")
        if partial_observation is None:
            # A noisy MDP observation or recurrent policy must not receive the
            # true automaton state. Preserve legacy fully-observed baselines
            # when the MDP observation channel is exact.
            partial_observation = (
                architecture == "gru" or self.mdp.observation_error > 0.0
            )
        partial_observation = bool(partial_observation)
        observation_factors = (
            (self.shape[0], self.mdp.shape[0], self.mdp.shape[1])
            if partial_observation else self.shape[:-1]
        )
        if architecture == "linear":
            # One independent output vector for each observation, or each
            # product state in the legacy fully-observed baseline.
            n_observations = int(np.prod(observation_factors))
            hidden_size = None
        elif architecture == "mlp":
            n_observations = sum(observation_factors)
            hidden_size = 64
        else:
            # GRU tokens contain the current observation, a valid-token bit,
            # and the preceding action. The history encoder adds a persistent
            # one-hot epsilon-target memory derived from the controller's own
            # selected edge, so branch identity survives beyond the k-step window.
            recurrent_token_size = sum(observation_factors) + 1 + n_actions
            epsilon_commitment_size = self.oa.shape[1]
            n_observations = recurrent_token_size + epsilon_commitment_size
            hidden_size = gru_hidden_size

        def new_network():
            if architecture == "gru":
                return dqn.GRUDQN(
                    n_observations, n_actions, hidden_size=gru_hidden_size
                ).to(device)
            return dqn.DQN(n_observations, n_actions,
                           hidden_size=hidden_size).to(device)

        policy_net = new_network()
        target_net = new_network()
        target_net.load_state_dict(policy_net.state_dict())
        target_net.eval()
        for parameter in target_net.parameters():
            parameter.requires_grad_(False)
        # A deterministic rejecting sink has zero continuation value: it has
        # no epsilon exits, every automaton edge returns to itself, and no
        # transition is accepting. Removing its bootstrap is Bellman-equivalent
        # to the tabular fixed point and prevents approximation error in the
        # sink from leaking backward into its predecessor states. Accepting
        # absorbing states do not meet this condition and keep bootstrapping.
        rejecting_sink_qs = {
            q for q in range(self.oa.shape[1])
            if not self.oa.eps[q]
            and self.oa.delta[q]
            and all(destination == q for destination in self.oa.delta[q].values())
            and not any(
                status is True
                for statuses in self.oa.acc[q].values()
                for status in statuses
            )
        }
        epsilon_sources = {
            q for q, outgoing in enumerate(self.oa.eps) if outgoing
        }
        epsilon_targets = {
            destination
            for outgoing in self.oa.eps
            for destination in outgoing
        }
        # For the safe-absorbing-states automaton, epsilon is available only
        # from q0, its targets have no further epsilon edges, and every
        # non-epsilon q0 transition either stays in q0 or reaches the rejecting
        # sink. The controller can therefore know epsilon availability from
        # its own action history without observing hidden q.
        epsilon_commit_mask_supported = (
            partial_observation
            and architecture == "gru"
            and bool(epsilon_targets)
            and epsilon_sources == {self.oa.q0}
            and all(not self.oa.eps[q] for q in epsilon_targets)
            and all(
                destination == self.oa.q0 or destination in rejecting_sink_qs
                for destination in self.oa.delta[self.oa.q0].values()
            )
        )
        optimizer = optim.AdamW(policy_net.parameters(), lr=learning_rate,
                                 amsgrad=True, weight_decay=0.0)
        memory = dqn.ReplayMemory(replay_capacity)
        criterion = nn.SmoothL1Loss()

        def encode_state(s, *, error_prob=None, previous_action=None):
            if partial_observation:
                observation = self.observe_observation(s, error_prob)
            else:
                observation = self.observe_state(s, error_prob)
            if architecture == "linear":
                index = np.ravel_multi_index(tuple(map(int, observation)),
                                             observation_factors)
                encoded = torch.zeros(n_observations, dtype=torch.float32,
                                      device=device)
                encoded[index] = 1.0
            elif architecture == "gru":
                encoded = self.encode_observation(
                    observation,
                    partial_observation=partial_observation,
                    previous_action=previous_action,
                    recurrent=True,
                ).to(device)
            else:
                encoded = torch.zeros(n_observations, dtype=torch.float32,
                                      device=device)
                offset = 0
                for coordinate, size in zip(observation, observation_factors):
                    encoded[offset + int(coordinate)] = 1.0
                    offset += size
            return encoded.reshape(1, -1)

        def observed_encoding(s, previous_action=None):
            return encode_state(s, previous_action=previous_action)

        def history_encoding(tokens, *, epsilon_target=None):
            """Pad tokens and append persistent epsilon-target memory."""
            tokens = list(tokens[-history_length:])
            if len(tokens) < history_length:
                padding = [torch.zeros(recurrent_token_size, dtype=torch.float32,
                                       device=device)
                           for _ in range(history_length - len(tokens))]
                tokens = padding + tokens
            encoded_history = torch.stack(tokens)
            valid_token_index = sum(observation_factors)
            valid_tokens = encoded_history[:, valid_token_index:valid_token_index + 1]
            target_features = torch.zeros(
                epsilon_commitment_size, dtype=torch.float32, device=device
            )
            if epsilon_target is not None:
                target_features[int(epsilon_target)] = 1.0
            commitment = valid_tokens * target_features.unsqueeze(0)
            encoded_history = torch.cat((encoded_history, commitment), dim=1)
            return encoded_history.unsqueeze(0)

        def observed_token(s, previous_action=None, *, error_prob=None):
            return encode_state(
                s, error_prob=error_prob, previous_action=previous_action
            ).squeeze(0)

        # For feed-forward networks, cache canonical (noise-free) encodings.
        # A recurrent policy instead builds a fresh history window per step.
        if architecture != "gru":
            encoded_states = {
                state: encode_state(state, error_prob=0.0)
                for state in self.states()
            }

        # The fallback action set is global, so it cannot reveal hidden q.
        # Some automata also permit a tighter mask derived only from the
        # controller's own epsilon-commitment history.
        global_epsilon_actions = sorted({
            len(self.mdp.A) + destination
            for outgoing in self.oa.eps
            for destination in outgoing
        })
        global_action_ids = list(range(len(self.mdp.A))) + global_epsilon_actions
        global_action_mask = torch.zeros(n_actions, dtype=torch.bool, device=device)
        global_action_mask[global_action_ids] = True
        mdp_action_ids = list(range(len(self.mdp.A)))
        mdp_action_mask = torch.zeros(n_actions, dtype=torch.bool, device=device)
        mdp_action_mask[mdp_action_ids] = True
        action_masks = {}
        for state in self.states():
            if partial_observation:
                action_masks[state] = global_action_mask
            else:
                mask = torch.zeros(n_actions, dtype=torch.bool, device=device)
                mask[self.A[state]] = True
                action_masks[state] = mask

        def get_action_mask(state, epsilon_committed=False):
            if not partial_observation:
                return action_masks[state]
            if epsilon_commit_mask_supported and epsilon_committed:
                return mdp_action_mask
            return global_action_mask
        action_tensors = [torch.tensor([[action]], dtype=torch.long, device=device)
                          for action in range(n_actions)]
        reward_tensors = {
            False: torch.tensor([0.0], dtype=torch.float32, device=device),
            True: torch.tensor([1.0 - self.discount_accepting],
                               dtype=torch.float32, device=device),
        }
        discount_tensors = {
            False: torch.tensor([self.discount_nonaccepting],
                                dtype=torch.float32, device=device),
            True: torch.tensor([self.discount_accepting],
                               dtype=torch.float32, device=device),
        }

        def optimize_model():
            if len(memory) < batch_size:
                return
            batch = Transition(*zip(*memory.sample(batch_size)))
            mask = torch.tensor([s is not None for s in batch.next_state],
                                device=device, dtype=torch.bool)
            next_states = [s for s in batch.next_state if s is not None]
            state_values = policy_net(torch.cat(batch.state)).gather(1, torch.cat(batch.action))
            next_values = torch.zeros(batch_size, device=device)
            if next_states:
                with torch.no_grad():
                    next_masks = torch.stack([m for m in batch.next_action_mask if m is not None])
                    # Double DQN: the online network selects the best legal
                    # next action; the target network evaluates that action.
                    # This reduces the positive bias from taking a noisy max.
                    online_next_q = policy_net(torch.cat(next_states))
                    online_next_q = online_next_q.masked_fill(~next_masks, -torch.inf)
                    next_actions = online_next_q.argmax(dim=1, keepdim=True)
                    target_next_q = target_net(torch.cat(next_states))
                    target_next_q = target_next_q.gather(1, next_actions).squeeze(1)
                    # True values are in [0,1] for this reward/discount pair.
                    # Project only the bootstrap target; leave online Q values
                    # unclipped so approximation errors remain observable.
                    next_values[mask] = target_next_q.clamp(0.0, 1.0)
            expected = torch.cat(batch.reward) + torch.cat(batch.discount) * next_values
            loss = criterion(state_values, expected.unsqueeze(1))
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 10.0)
            optimizer.step()

        episode_returns = []
        start_states = self.start_states()
        if not start_states:
            raise ValueError("Cannot train: the grid has no start states.")
        total_steps = 0
        transition_counts = np.zeros(self.shape, dtype=np.int64)
        log_every = max(1, K // 100)
        from datetime import datetime, timezone
        import hashlib
        from pathlib import Path
        import subprocess
        import time
        run_started_at = datetime.now(timezone.utc)
        run_started_perf = time.perf_counter()
        for k in range(K):
            if k % log_every == 0:
                if episode_returns:
                    return_window = min(1000, len(episode_returns))
                    recent_return = float(
                        np.mean(episode_returns[-return_window:])
                    )
                    print(
                        f"Episode {k}/{K}; trailing {return_window}-episode "
                        f"return={recent_return:.6f}"
                    )
                else:
                    print(f"Episode {k}/{K}")
            mdp_start = (start if start is not None else
                         start_states[np.random.randint(len(start_states))])
            state = (self.shape[0]-1, self.oa.q0) + mdp_start
            epsilon_committed = False
            epsilon_target = None
            if architecture == "gru":
                history_tokens = [observed_token(state)]
                st = history_encoding(
                    history_tokens, epsilon_target=epsilon_target
                )
            else:
                st = observed_encoding(state)
            # The paper anneals epsilon from 1.0 to 0.1. The separate
            # automaton-action probability improves coverage of legal
            # epsilon branches while leaving replay sampling uniform.
            epsilon = max(
                1.0 - (1.0 - epsilon_floor) * k / max(1, K - 1),
                epsilon_floor,
            )
            episode_return = 0.0
            return_discount = 1.0
            for t in range(T):
                current_action_mask = get_action_mask(state, epsilon_committed)
                legal = [
                    action for action in range(n_actions)
                    if current_action_mask[action]
                ]
                automaton_actions = [
                    action for action in legal if action >= len(self.mdp.A)
                ]
                if (automaton_actions
                        and random.random() < automaton_action_probability):
                    action = random.choice(automaton_actions)
                elif random.random() < epsilon:
                    action = random.choice(legal)
                else:
                    with torch.no_grad():
                        v = policy_net(st).squeeze(0)
                        v = v.masked_fill(~current_action_mask, -torch.inf)
                    action = int(v.argmax().item())
                transition_counts[state + (action,)] += 1
                if action in self.A[state]:
                    states, probs = self.transition_probs[state][action]
                elif partial_observation and action in global_epsilon_actions:
                    # The controller cannot know whether an epsilon edge is
                    # enabled. An unavailable attempt is a no-op, keeping a
                    # fixed observation-independent action set.
                    states, probs = [state], [1.0]
                else:
                    raise RuntimeError(f"Action {action} is invalid in state {state}")
                next_state = states[np.random.choice(len(states), p=probs)]
                next_epsilon_committed = (
                    epsilon_committed or action in global_epsilon_actions
                )
                next_epsilon_target = epsilon_target
                if action in global_epsilon_actions and next_epsilon_target is None:
                    next_epsilon_target = action - len(self.mdp.A)
                if architecture == "gru":
                    next_token = observed_token(next_state, previous_action=action)
                    next_history_tokens = (
                        history_tokens + [next_token]
                    )[-history_length:]
                    next_st = history_encoding(
                        next_history_tokens,
                        epsilon_target=next_epsilon_target,
                    )
                else:
                    next_st = observed_encoding(next_state)
                    next_history_tokens = None
                # A proven zero-value sink can terminate the Bellman target
                # without exposing q in the policy input or action mask. The
                # simulator rollout itself continues unchanged.
                next_is_rejecting_sink = next_state[1] in rejecting_sink_qs
                reward = float(self.reward[state])
                # Use None only in the replay target for a proven zero-value
                # sink. The rollout itself continues so this changes no reward
                # semantics or accepting-state behavior.
                replay_next_st = None if next_is_rejecting_sink else next_st
                next_mask = (
                    None if next_is_rejecting_sink
                    else get_action_mask(next_state, next_epsilon_committed)
                )
                # Match tabular Q-learning: fixed horizon, but bootstrap on every step.
                # Use the same reward-conditioned discount as tabular Q-learning.
                transition_discount = (self.discount_accepting if reward
                                       else self.discount_nonaccepting)
                episode_return += return_discount * reward
                return_discount *= transition_discount
                reward_event = reward != 0
                if state[1] not in rejecting_sink_qs:
                    memory.push(
                        st,
                        action_tensors[action],
                        replay_next_st,
                        next_mask,
                        reward_tensors[reward_event],
                        discount_tensors[reward_event],
                    )
                state = next_state
                st = next_st
                if architecture == "gru":
                    history_tokens = next_history_tokens
                epsilon_committed = next_epsilon_committed
                epsilon_target = next_epsilon_target
                total_steps += 1
                if total_steps % update_every == 0:
                    optimize_model()
                if total_steps % target_update_every == 0:
                    target_net.load_state_dict(policy_net.state_dict())
            episode_returns.append(episode_return)

        # Keep returns available for notebook inspection without opening a
        # plotting window. A compact text summary is useful during long runs;
        # the full per-episode series is also included in saved checkpoints.
        self.last_episode_returns = np.asarray(episode_returns, dtype=np.float64)
        if self.last_episode_returns.size:
            summary_window = min(1000, self.last_episode_returns.size)
            self.last_training_return_summary = {
                "episodes": int(self.last_episode_returns.size),
                "mean": float(self.last_episode_returns.mean()),
                "std": float(self.last_episode_returns.std()),
                "first_window_mean": float(
                    self.last_episode_returns[:summary_window].mean()
                ),
                "last_window_mean": float(
                    self.last_episode_returns[-summary_window:].mean()
                ),
                "window_episodes": int(summary_window),
            }
            print(
                "Training returns: "
                f"mean={self.last_training_return_summary['mean']:.6f}, "
                f"first {summary_window} mean="
                f"{self.last_training_return_summary['first_window_mean']:.6f}, "
                f"last {summary_window} mean="
                f"{self.last_training_return_summary['last_window_mean']:.6f}"
            )
        else:
            self.last_training_return_summary = {
                "episodes": 0,
                "mean": None,
                "std": None,
                "first_window_mean": None,
                "last_window_mean": None,
                "window_episodes": 0,
            }

        Q = np.full(self.shape, -np.inf, dtype=np.float32)
        with torch.no_grad():
            for state in self.states():
                # A recurrent Q value depends on history. The table returned
                # here uses a cold-start history containing only this clean
                # current observation; use the returned model for live policy
                # decisions with the actual observation history.
                if architecture == "gru":
                    token = observed_token(state, error_prob=0.0)
                    network_input = history_encoding([token])
                else:
                    network_input = encoded_states[state]
                Q[state] = policy_net(network_input).squeeze(0).cpu().numpy()
                valid_actions = (global_action_ids if partial_observation
                                 else self.A[state])
                Q[state][list(set(range(n_actions)) - set(valid_actions))] = -np.inf
                if not partial_observation and state[1] in rejecting_sink_qs:
                    Q[state][self.A[state]] = 0.0

        if checkpoint_dir is not None:
            import json

            project_dir = Path(__file__).resolve().parent
            checkpoint_root = Path(checkpoint_dir).expanduser()
            if not checkpoint_root.is_absolute():
                checkpoint_root = project_dir / checkpoint_root
            checkpoint_root.mkdir(parents=True, exist_ok=True)

            def git_output(*args):
                try:
                    return subprocess.check_output(
                        ["git", *args], cwd=project_dir, text=True,
                        stderr=subprocess.DEVNULL,
                    ).strip()
                except (OSError, subprocess.CalledProcessError):
                    return None

            git_commit = git_output("rev-parse", "HEAD")
            git_status = git_output("status", "--porcelain", "--untracked-files=all")
            dirty_files = (
                [line[3:] for line in git_status.splitlines() if line]
                if git_status is not None else None
            )
            source_hashes = {}
            for source_name in ("controlsyn.py", "dqn.py"):
                source_path = project_dir / source_name
                if source_path.exists():
                    source_hashes[source_name] = hashlib.sha256(
                        source_path.read_bytes()
                    ).hexdigest()

            run_finished_at = datetime.now(timezone.utc)
            timestamp = run_started_at.strftime("%Y%m%dT%H%M%S.%fZ")
            commit_tag = git_commit[:7] if git_commit else "nogit"
            run_id = f"{timestamp}_K{K}_T{T}_{commit_tag}"
            metadata = {
                "schema_version": 1,
                "run_id": run_id,
                "started_at_utc": run_started_at.isoformat(),
                "finished_at_utc": run_finished_at.isoformat(),
                "elapsed_seconds": time.perf_counter() - run_started_perf,
                "episodes": int(K),
                "steps_per_episode": int(T),
                "environment_steps": int(total_steps),
                "training_return_summary": self.last_training_return_summary,
                "start_mdp_square": list(map(int, start)) if start is not None else None,
                "start_sampling": (
                    "uniform over grid states" if start is None else "fixed square"
                ),
                "epsilon_schedule": "linear from 1.0 to epsilon_floor",
                "epsilon_floor": float(epsilon_floor),
                "learning_rate": float(learning_rate),
                "automaton_action_probability": float(automaton_action_probability),
                "architecture": architecture,
                "n_observations": int(n_observations),
                "n_actions": int(n_actions),
                "hidden_size": hidden_size,
                "partial_observation": partial_observation,
                "observation_encoding": (
                    "pair,row,column; automaton state hidden"
                    if partial_observation else
                    "pair,automaton state,row,column"
                ),
                "history_length": int(history_length) if architecture == "gru" else None,
                "gru_hidden_size": int(gru_hidden_size) if architecture == "gru" else None,
                "recurrent_token_includes_previous_action": architecture == "gru",
                "recurrent_input_includes_epsilon_commitment": architecture == "gru",
                "recurrent_epsilon_commitment_mode": (
                    "target_one_hot" if architecture == "gru" else "none"
                ),
                "recurrent_epsilon_commitment_size": (
                    int(epsilon_commitment_size) if architecture == "gru" else 0
                ),
                "observation_error": float(self.mdp.observation_error),
                "recurrent_padding": "left zero padding with valid-token bit",
                "partial_action_semantics": (
                    "grid moves plus epsilon edges until an observed epsilon commitment; then grid moves only"
                    if epsilon_commit_mask_supported else
                    "all grid moves and epsilon edges; unavailable epsilon is no-op"
                    if partial_observation else None
                ),
                "epsilon_commit_mask_from_history": epsilon_commit_mask_supported,
                "rejecting_sink_target_semantics": (
                    "zero bootstrap on proven rejecting-sink transitions; hidden q is not a policy input"
                ),
                "q_table_semantics": (
                    "cold-start history with one clean current observation"
                    if architecture == "gru" else None
                ),
                "replay_capacity": int(replay_capacity),
                "batch_size": int(batch_size),
                "update_every": int(update_every),
                "target_update_every": int(target_update_every),
                "discount_nonaccepting": float(self.discount_nonaccepting),
                "discount_accepting": float(self.discount_accepting),
                "product_state_shape": list(map(int, self.shape[:-1])),
                "mdp_shape": list(map(int, self.mdp.shape)),
                "automaton_shape": list(map(int, self.oa.shape)),
                "automaton_initial_state": int(self.oa.q0),
                "gru_num_layers": 1 if architecture == "gru" else None,
                "git_commit": git_commit,
                "git_dirty_files": dirty_files,
                "source_sha256": source_hashes,
            }
            checkpoint_path = checkpoint_root / f"dqn_{run_id}.pt"
            checkpoint_tmp = checkpoint_path.with_suffix(".pt.tmp")
            checkpoint = {
                "metadata": metadata,
                "policy_state_dict": {
                    key: value.detach().cpu()
                    for key, value in policy_net.state_dict().items()
                },
                "target_state_dict": {
                    key: value.detach().cpu()
                    for key, value in target_net.state_dict().items()
                },
                "optimizer_state_dict": optimizer.state_dict(),
                "q_values": torch.from_numpy(Q.copy()),
                "episode_returns": torch.from_numpy(
                    self.last_episode_returns.copy()
                ),
            }
            torch.save(checkpoint, checkpoint_tmp)
            os.replace(checkpoint_tmp, checkpoint_path)

            # Keep a readable metadata sidecar next to the full PyTorch checkpoint.
            metadata_path = checkpoint_path.with_suffix(".json")
            metadata_tmp = metadata_path.with_suffix(".json.tmp")
            with metadata_tmp.open("w", encoding="utf-8") as metadata_file:
                json.dump(metadata, metadata_file, indent=2, sort_keys=True)
                metadata_file.write("\n")
            os.replace(metadata_tmp, metadata_path)
            self.last_checkpoint_path = str(checkpoint_path)
            print(f"Saved DQN checkpoint: {checkpoint_path}")
        if return_diagnostics:
            if return_model:
                return Q, transition_counts, policy_net
            return Q, transition_counts
        if return_model:
            return Q, policy_net
        return Q
