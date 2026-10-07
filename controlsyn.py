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
        """Return the learner's noisy observation of a product state."""
        i, q, r, c = state
        observed_r, observed_c = self.mdp.observe_state((r, c), error_prob)
        return (i, q, observed_r, observed_c)
    
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
        value = np.zeros(self.shape[:-1])
        old_value = np.copy(value)
        t = 0  # The time step
        d = np.inf  # The difference between the last two steps
        while (T and t<T) or (threshold and d>threshold):
            value, old_value = old_value, value
            for state in self.states():
                # Bellman operator
                action_values = np.empty(len(self.A[state]))
                for i,action in enumerate(self.A[state]):
                    action_values[i] = np.sum([old_value[s]*p for s,p in zip(*self.transition_probs[state][action])])
                gamma = (self.discount_accepting if self.reward[state] > 0
                         else self.discount_nonaccepting)
                value[state] = self.reward[state] + gamma*np.max(action_values)
            t += 1
            d = np.nanmax(np.abs(old_value-value))
            
        return value
    
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

    def deep_q_learning(self, start=None, T=None, K=None):
        import random
        import dqn
        from dqn import Transition
        device = torch.device("cuda" if torch.cuda.is_available() else
                              "mps" if torch.backends.mps.is_available() else "cpu")
        # T is a rollout/reset horizon, not a terminal condition. Grid traps
        # continue to accrue accepting rewards and must not be marked done.
        T = T if T else int(np.prod(self.shape[:-1]))
        K = K if K else 100000
        batch_size = 64
        update_every = 10
        target_update_every = 1_000  # Hard target-network syncs in environment steps.
        n_actions = self.shape[-1]
        # Encode the four product-state components as separate one-hot blocks:
        # Rabin pair, automaton state, row, and column.
        n_observations = sum(self.shape[:-1])
        # A prior run collapsed distinct product states toward one value.
        # Thirty-two units per layer give the factored encoding more capacity
        # to separate accepting and rejecting regions.
        policy_net = dqn.DQN(n_observations, n_actions, hidden_size=32).to(device)
        target_net = dqn.DQN(n_observations, n_actions, hidden_size=32).to(device)
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
        optimizer = optim.AdamW(policy_net.parameters(), lr=1e-4, amsgrad=True, weight_decay=0.0)
        memory = dqn.ReplayMemory(20_000)
        criterion = nn.SmoothL1Loss()

        def encode(s):
            encoded = torch.zeros((1, n_observations),
                                  dtype=torch.float32, device=device)
            offset = 0
            for coordinate, size in zip(s, self.shape[:-1]):
                encoded[0, offset + int(coordinate)] = 1.0
                offset += size
            return encoded

        def observed_encoding(s):
            return encoded_states[self.observe_state(s)]

        # These state and action encodings are immutable during training, so
        # build them once instead of allocating device tensors on every step.
        encoded_states = {state: encode(state) for state in self.states()}
        action_masks = {}
        for state in self.states():
            mask = torch.zeros(n_actions, dtype=torch.bool, device=device)
            mask[self.A[state]] = True
            action_masks[state] = mask
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
            # Uniform replay avoids overweighting rare accepting rewards in
            # the shared function approximator.
            batch = Transition(*zip(*memory.sample(
                batch_size, reward_fraction=0.0, epsilon_fraction=0.0)))
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
        log_every = max(1, K // 100)
        for k in range(K):
            if k % log_every == 0:
                print(f"Episode {k}/{K}")
            mdp_start = (start if start is not None else
                         start_states[np.random.randint(len(start_states))])
            state = (self.shape[0]-1, self.oa.q0) + mdp_start
            st = observed_encoding(state)
            # The paper anneals exploration from 1.0 to 0.1. Keep that floor
            # so the nondeterministic epsilon choices remain discoverable.
            epsilon = max(1.0 - 0.9*k/max(1, K - 1), 0.1)
            episode_return = 0.0
            return_discount = 1.0
            for t in range(T):
                legal = self.A[state]
                if random.random() < epsilon:
                    action = random.choice(legal)
                else:
                    with torch.no_grad():
                        v = policy_net(st).squeeze(0)
                        v = v.masked_fill(~action_masks[state], -torch.inf)
                    action = int(v.argmax().item())
                states, probs = self.transition_probs[state][action]
                next_state = states[np.random.choice(len(states), p=probs)]
                next_st = observed_encoding(next_state)
                next_is_rejecting_sink = next_state[1] in rejecting_sink_qs
                reward = float(self.reward[state])
                # Use None only in the replay target for a proven zero-value
                # sink. The rollout itself continues so this changes no reward
                # semantics or accepting-state behavior.
                replay_next_st = None if next_is_rejecting_sink else next_st
                next_mask = None if next_is_rejecting_sink else action_masks[next_state]
                # Match tabular Q-learning: fixed horizon, but bootstrap on every step.
                # Use the same reward-conditioned discount as tabular Q-learning.
                transition_discount = (self.discount_accepting if reward
                                       else self.discount_nonaccepting)
                episode_return += return_discount * reward
                return_discount *= transition_discount
                reward_event = reward != 0
                memory.push(
                    st,
                    action_tensors[action],
                    replay_next_st,
                    next_mask,
                    reward_tensors[reward_event],
                    discount_tensors[reward_event],
                    reward_event=reward_event,
                    epsilon_action=(action >= len(self.mdp.A)),
                )
                state = next_state
                st = next_st
                total_steps += 1
                if total_steps % update_every == 0:
                    optimize_model()
                if total_steps % target_update_every == 0:
                    target_net.load_state_dict(policy_net.state_dict())
            episode_returns.append(episode_return)

        if 'plt' in globals():
            fig, ax = plt.subplots(figsize=(8, 4))
            episodes = np.arange(1, len(episode_returns) + 1)
            ax.plot(episodes, episode_returns, alpha=0.25, label='Episode return')
            window = min(50, len(episode_returns))
            if window > 1:
                moving_average = np.convolve(
                    episode_returns, np.ones(window) / window, mode='valid')
                ax.plot(episodes[window - 1:], moving_average,
                        label=f'{window}-episode average')
            ax.set_xlabel('Episode')
            ax.set_ylabel('Return')
            ax.set_title('DQN training return')
            ax.grid(alpha=0.25)
            ax.legend()
            fig.tight_layout()
            plt.show()

        Q = np.full(self.shape, -np.inf, dtype=np.float32)
        with torch.no_grad():
            for state in self.states():
                # Keep the returned table deterministic. During training the
                # network receives noisy observations; the public Q-table is
                # indexed by the corresponding clean product state.
                Q[state] = policy_net(encoded_states[state]).squeeze(0).cpu().numpy()
                Q[state][list(set(range(n_actions)) - set(self.A[state]))] = -np.inf
        return Q
