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
        The reward function of the star-MDP. self.reward[state] = 1-discountB if 'state' belongs to B, 0 otherwise.
        
    transition_probs : array, shape=(n_pairs,n_qs,n_rows,n_cols,n_actions)
        The transition probabilities. self.transition_probs[state][action] stores a pair of lists ([s1,s2,..],[p1,p2,...]) that contains only positive probabilities and the corresponding transitions.
    
    Parameters
    ----------
    mdp : mdp.GridMDP
        The MDP that models the environment.
        
    oa : oa.OmegaAutomatan
        The OA obtained from the LTL specification.
        
    discount : float
        The discount factor.
    
    discountB : float
        The discount factor applied to B states.
    
    """
    def __init__(self, mdp, oa, discount=0.99999, discountB=0.99):
        self.mdp = mdp
        self.oa = oa
        self.discount = discount
        self.discountB = discountB  # We can also explicitly define a function of discount
        self.shape = oa.shape + mdp.shape + (len(mdp.A)+oa.shape[1],)
        
        # Create the action matrix
        self.A = np.empty(self.shape[:-1],dtype=object)
        for i,q,r,c in self.states():
            self.A[i,q,r,c] = list(range(len(mdp.A))) + [len(mdp.A)+e_a for e_a in oa.eps[q]]
        
        # Create the reward matrix
        self.reward = np.zeros(self.shape[:-1])
        for i,q,r,c in self.states():
            self.reward[i,q,r,c] = 1-self.discountB if oa.acc[q][mdp.label[r,c]][i] else 0
        
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
        
        T = T if T else np.prod(self.shape[:-1])
        K = K if K else 100000
        
        Q = np.zeros(self.shape)

        for k in range(K+1):
            if((k%2500) == 0): print(k)
            state = (self.shape[0]-1,self.oa.q0)+(start if start else self.mdp.random_state())
            alpha = np.max((1.0*(1 - 1.5*k/K),0.001))
            epsilon = np.max((1.0*(1 - 1.5*k/K),0.01))
            for t in range(T):

                reward = self.reward[state]
                gamma = self.discountB if reward else self.discount
                
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
                gamma = self.discountB if self.reward[state]>0 else self.discount
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
        T = T if T else int(np.prod(self.shape[:-1]))
        K = K if K else 100000
        gamma, batch_size, tau = 0.99, 20, 0.005
        update_every = 10
        n_actions = self.shape[-1]
        policy_net = dqn.DQN(4, n_actions).to(device)
        target_net = dqn.DQN(4, n_actions).to(device)
        target_net.load_state_dict(policy_net.state_dict())
        optimizer = optim.AdamW(policy_net.parameters(), lr=3e-4, amsgrad=True)
        memory = dqn.ReplayMemory(10000)
        criterion = nn.SmoothL1Loss()

        def encode(s):
            i, q, r, c = s
            ni, nq, nr, nc, _ = self.shape
            return torch.tensor([[i/max(1, ni-1), q/max(1, nq-1),
                                  r/max(1, nr-1), c/max(1, nc-1)]],
                                dtype=torch.float32, device=device)

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
                    next_q = target_net(torch.cat(next_states))
                    next_masks = torch.stack([m for m in batch.next_action_mask if m is not None])
                    next_q = next_q.masked_fill(~next_masks, -torch.inf)
                    next_values[mask] = next_q.max(1).values
            expected = torch.cat(batch.reward) + torch.cat(batch.discount) * next_values
            loss = criterion(state_values, expected.unsqueeze(1))
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
            optimizer.step()
            with torch.no_grad():
                for tp, pp in zip(target_net.parameters(), policy_net.parameters()):
                    tp.mul_(1-tau).add_(tau*pp)

        episode_returns = []
        for k in range(K + 1):
            if(k % 100 == 0): print(f"Episode {k}")
            state = (self.shape[0]-1, self.oa.q0) + (start if start else self.mdp.random_state())
            epsilon = max(1 - 1.5*k/max(1, K), 0.01)
            episode_return = 0.0
            for t in range(T):
                st = encode(state); legal = list(self.A[state])
                if random.random() < epsilon:
                    action = random.choice(legal)
                else:
                    with torch.no_grad():
                        v = policy_net(st).squeeze(0)
                    action = max(legal, key=lambda a: v[a].item())
                states, probs = self.transition_probs[state][action]
                next_state = states[np.random.choice(len(states), p=probs)]
                reward = float(self.reward[state])
                next_mask = torch.zeros(n_actions, dtype=torch.bool, device=device)
                next_mask[list(self.A[next_state])] = True
                # Match tabular Q-learning: fixed horizon, but bootstrap on every step.
                transition_discount = self.discountB if reward != 0 else self.discount
                memory.push(
                    st,
                    torch.tensor([[action]], dtype=torch.long, device=device),
                    encode(next_state),
                    next_mask,
                    torch.tensor([reward], dtype=torch.float32, device=device),
                    torch.tensor([transition_discount], dtype=torch.float32, device=device),
                )
                episode_return += reward
                if (k * T + t + 1) % update_every == 0:
                    optimize_model()
                state = next_state
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
                Q[state] = policy_net(encode(state)).squeeze(0).cpu().numpy()
                Q[state][list(set(range(n_actions)) - set(self.A[state]))] = -np.inf
        return Q


