"""Control Synthesis using Reinforcement Learning.
"""
import numpy as np
from itertools import product
from mdp import GridMDP
import os
import importlib

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

    def reinforce(self, T=None, K=None):
        # preference table is H
        T = T if T else np.prod(self.shape[:-1])
        K = K if K else 100000
        alpha = 0.01
                
        H = np.zeros(self.shape)
        
        for k in range(K+1):
            if((k%2500) == 0): print(k)
            state = (self.shape[0]-1,self.oa.q0)+self.mdp.random_state()
            trajectory = [] # (s, a, r, pi(a|s))
            for t in range(T):

                # create pi(* | s) from soft max of H values
                pi = dict() # (action, probability)
                actions = self.A[state]
                for action in actions:
                    pi[action] = H[state][action]
                vals = {a: H[state][a] for a in actions}
                max_val = max(vals.values())
                exp_vals = {a: np.exp(vals[a] - max_val) for a in vals}
                sum_h = sum(exp_vals.values())
                pi = {a: exp_vals[a] / sum_h for a in exp_vals}
                    
                chosen_action = list(pi.keys())[np.random.choice(len(actions), p=list(pi.values()))]
                reward = self.reward[state]
                trajectory.append((state, chosen_action, reward, pi))

                # Observe the next state
                states, probs = self.transition_probs[state][chosen_action]
                next_state = states[np.random.choice(len(states),p=probs)]

                state = next_state

            G = 0 # return with discount
            t = T-1
            while(t >= 0):
                (state, action, reward, pi_state) = trajectory[t]
                gamma = self.discountB if reward > 0 else self.discount
                G = reward + gamma * G 

                actions = self.A[state]
                for possible_action in actions:
                    indicator = 1 if possible_action == action else 0
                    H[state][possible_action] = H[state][possible_action] + alpha * G * (indicator - pi_state[possible_action])
                

                t -= 1

        return H

    def policy_from_H(self, H):
        policy = np.zeros(self.shape[:-1], dtype=int)
        for state in self.states():
            actions = self.A[state]
            prefs = [H[state][a] for a in actions]
            policy[state] = actions[np.argmax(prefs)]
        return policy

        
    
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


    def compare_learning(self, checkpoints=(100, 1000, 5000, 10000),
                         T=None, eval_episodes=100, seed=None, plot=True):
        """Compare Q-learning and REINFORCE at training checkpoints.

        Each checkpoint is trained from a fresh zero-initialized learner, so
        the reported curves show performance as a function of total training
        episodes.  Performance is estimated by following the greedy policy
        and reporting mean undiscounted return and the fraction of evaluation
        episodes that finish in an accepting product state.

        Returns a dictionary with arrays keyed by ``episodes``, ``q_return``,
        ``reinforce_return``, ``q_success`` and ``reinforce_success``.
        """
        checkpoints = np.asarray(checkpoints, dtype=int)
        if checkpoints.ndim != 1 or np.any(checkpoints <= 0):
            raise ValueError("checkpoints must be a 1-D sequence of positive integers")
        if np.any(np.diff(checkpoints) < 0):
            raise ValueError("checkpoints must be sorted")
        horizon = int(T if T is not None else np.prod(self.shape[:-1]))
        rng_state = np.random.get_state()

        def evaluate(policy):
            returns, successes = [], []
            for _ in range(eval_episodes):
                state = (self.shape[0] - 1, self.oa.q0) + self.mdp.random_state()
                total = 0.0
                accepted = False
                for _ in range(horizon):
                    reward = self.reward[state]
                    # Do not discount evaluation rewards: this metric is
                    # intended to measure raw task performance.
                    total += reward
                    accepted = accepted or reward > 0
                    states, probs = self.transition_probs[state][policy[state]]
                    state = states[np.random.choice(len(states), p=probs)]
                returns.append(total)
                successes.append(accepted)
            return float(np.mean(returns)), float(np.mean(successes))

        def q_policy_from_table(q):
            """Extract the greedy policy using only legal Q-table actions."""
            policy = np.zeros(self.shape[:-1], dtype=int)
            for state in self.states():
                legal = self.A[state]
                policy[state] = legal[np.argmax([q[state][a] for a in legal])]
            return policy

        results = {k: [] for k in
                   ('episodes', 'q_return', 'reinforce_return',
                    'q_success', 'reinforce_success')}
        try:
            for episodes in checkpoints:
                if seed is not None:
                    np.random.seed(seed)
                q = self.q_learning(T=horizon, K=int(episodes))
                q_policy = q_policy_from_table(q)
                q_ret, q_ok = evaluate(q_policy)

                if seed is not None:
                    np.random.seed(seed)
                h = self.reinforce(T=horizon, K=int(episodes))
                r_policy = self.policy_from_H(h)
                r_ret, r_ok = evaluate(r_policy)

                results['episodes'].append(int(episodes))
                results['q_return'].append(q_ret)
                results['reinforce_return'].append(r_ret)
                results['q_success'].append(q_ok)
                results['reinforce_success'].append(r_ok)
        finally:
            np.random.set_state(rng_state)

        if plot:
            fig, axes = plt.subplots(1, 2, figsize=(11, 4))
            x = results['episodes']
            axes[0].plot(x, results['q_return'], marker='o', label='Q-learning')
            axes[0].plot(x, results['reinforce_return'], marker='o', label='REINFORCE')
            axes[0].set(xlabel='training episodes', ylabel='mean undiscounted reward')
            axes[1].plot(x, results['q_success'], marker='o', label='Q-learning')
            axes[1].plot(x, results['reinforce_success'], marker='o', label='REINFORCE')
            axes[1].set(xlabel='training episodes', ylabel='fraction reaching acceptance')
            for ax in axes:
                ax.grid(alpha=.3)
                ax.legend()
            fig.tight_layout()
            results['figure'] = fig
        return results
