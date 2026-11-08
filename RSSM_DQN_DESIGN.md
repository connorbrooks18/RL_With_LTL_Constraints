# Recurrent state-space representation feeding a DQN

## Proposal

Keep the current off-policy DQN and replace its four-observation GRU encoder
with an RSSM encoder. The RSSM learns a compact latent state from observation
and action sequences. The existing Double-DQN target, replay-based TD loss,
action masks, epsilon exploration, and Q-learning policy remain in place.

This is an adaptation of RSSM world models, not Dreamer as a whole. PlaNet
introduced a latent dynamics model with deterministic and stochastic
transition components. DreamerV2 learns behavior in a discrete latent world
model, but uses an actor-critic rather than a DQN. Here, the world-model
prediction losses train the representation, and the DQN remains responsible
for action values and control. See [PlaNet](https://arxiv.org/abs/1811.04551)
and [DreamerV2](https://arxiv.org/abs/2010.02193).

## Candidate model

At step `t`, use the visible observation `o_t = (task-pair, noisy-row,
noisy-column)`, previous action `a_(t-1)`, and the controller's remembered
epsilon target. The true automaton state remains hidden.

```text
h_t = GRUCell([z_(t-1), onehot(a_(t-1)), epsilon-memory], h_(t-1))
prior_t     = p(z_t | h_t)
posterior_t = q(z_t | h_t, encode(o_t))
phi_t = [h_t, posterior_t representation]
Q(phi_t, ·) = existing DQN Q head
```

The prior predicts the latent state before seeing the current observation; the
posterior incorporates that observation. Use a small discrete stochastic
latent, such as a product of categorical variables, alongside the deterministic
GRU state. Start with a compact configuration, then compare latent size as a
separate ablation. Feed the Q head the deterministic state plus posterior
features (for example, categorical probabilities), so it can use uncertainty
from aliased observations instead of seeing only one sampled latent.

Train the representation with an auxiliary world-model objective in addition
to the DQN loss:

```text
L = L_DQN
  + λ_obs L_observation_prediction
  + λ_reward L_reward_prediction
  + β KL(posterior_t || prior_t)
```

Predict the observation and reward from the latent state. Do not use the hidden
automaton state or true position as an RSSM input or training target; that
would leak privileged state into a partially observable experiment. A
continuation prediction can be added if the task has meaningful terminal
states, but the current 600-step episode boundary is a rollout horizon and the
Bellman target continues to bootstrap.

## Replay and target-network changes

The current replay stores independent transitions whose states already
contain a four-token history ([ReplayMemory in `dqn.py`](dqn.py) and transition
construction in [`controlsyn.py`](controlsyn.py)). An RSSM carries latent
memory across time, so independent transitions are insufficient. Store
episode-aligned sequences of observations, actions, rewards, next observations,
action masks, reset markers, and epsilon-memory updates. Train from contiguous
chunks with a burn-in prefix to reconstruct the latent state before applying
the DQN loss. Reset RSSM state at environment episode boundaries.

The target branch must use a target RSSM and target Q head. Build online and
target latent features consistently for Double DQN selection and evaluation;
sync the complete target model at the existing target-update cadence. Test
representation drift and Q-target stability before tuning the world-model
loss weights.

## Training schedule and comparison

The first pilot should use one matched seed and one environment at a time. Keep
the task, partial-observation channel, 600-step episodes, replay update rate,
target-update cadence, exploration schedule, evaluation starts, and DQN loss
matched to the history-4 GRU-DQN baseline. Record both sample count and wall
time because RSSM sequence updates add compute.

For the first comparison, evaluate these two systems:

1. Existing history-4 GRU-DQN.
2. RSSM-DQN: same Q-learning system, with the RSSM representation and auxiliary
   prediction losses described above.

Do not add imagined rollouts, a policy actor, or an actor-critic in this first
test. Those would change the control algorithm and make it unclear whether a
gain came from the representation or from planning. If the RSSM-DQN pilot is
competitive, then evaluate the same freeze schedule used in the current
experiment: train the RSSM and Q head jointly, freeze the RSSM, and continue
updating only the Q head. This directly tests whether the more expensive
representation can also produce a reusable latent state for a cheaper DQN
head phase.

Use fixed common evaluation scenarios. Report task return, policy maps over
the grid, observation/reward prediction accuracy, KL magnitude, training wall
time, and the exact episode at which the recurrent representation is frozen.
The representation metrics are diagnostics; policy return and wall time stay
the primary outcomes.

## Main risks

- **More compute:** the RSSM adds sequence training and prediction losses, so
  it may take longer per episode than the current GRU-DQN even if its latent
  state is better.
- **Representation drift:** Q values change as the encoder and latent dynamics
  change. A target RSSM, sequence burn-in, and delayed target updates are
  important for stable TD targets.
- **Wrong prediction target:** reconstructing noisy observations can spend
  capacity on sensor noise, while predicting hidden states would leak labels.
  Compare observation/reward prediction with task return and do not optimize
  reconstruction accuracy alone.
- **No guaranteed gain over a GRU:** DreamerV3's RSSM still uses a recurrent
  sequence model. The meaningful addition here is stochastic latent dynamics
  and predictive representation training, not the removal of recurrence.

## Local implementation touchpoints

- [`dqn.py`](dqn.py): `GRUDQN` and `ReplayMemory`.
- [`controlsyn.py`](controlsyn.py): partial observation encoding, episode
  rollout, transition insertion, Double-DQN loss, target-network update.
- [`train_pomdp.py`](train_pomdp.py): experiment arguments and model selection.
- [`evaluate_pomdp_vs_value_iteration.py`](evaluate_pomdp_vs_value_iteration.py):
  common-scenario policy evaluation.

