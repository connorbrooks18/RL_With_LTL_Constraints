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
h_(1:T)     = fused_GRU(encode(o_(1:T), previous actions, epsilon-memory))
prior_t     = p(z_t | h_t)
posterior_t = q(z_t | h_t, encode(o_t))
phi_t = [h_t, posterior_t representation]
Q(phi_t, ·) = existing DQN Q head
```

The prior predicts the latent state before seeing the current observation; the
posterior incorporates that observation. The pilot uses a compact 16-unit
deterministic state and two 4-way categorical variables. Feed the Q head the
deterministic state plus posterior probabilities (24 features total), so it can
use uncertainty from aliased observations instead of seeing only one sampled
latent. The implementation uses posterior expectations as the Q input. It
learns prior/posterior categorical states, but does not feed the stochastic
state back into the deterministic transition; the fused GRU reduces the
per-step-cell overhead. This is an RSSM-inspired encoder approximation, not a
full canonical RSSM.

Train the representation with an auxiliary world-model objective in addition
to the DQN loss:

```text
L = L_DQN
  + λ_obs L_observation_prediction
  + λ_reward L_reward_prediction
  + β KL(posterior_t || prior_t)
```

Predict the next partial observation and current reward event from the latent
state and chosen action. Do not use the hidden automaton state or true position
as an RSSM input or training target; that would leak privileged state into a
partially observable experiment. The 600-step episode boundary is a rollout
horizon and the Bellman target continues to bootstrap.

## Replay and target-network changes

The pilot runs the RSSM over the same four-token history already stored in
each replay transition. This keeps independent-transition replay and matches
the observation context available to the history-4 GRU. The recurrent
deterministic and stochastic state resets at the start of each history window;
it does not retain information older than four observations. A later full
RSSM study could use episode-aligned chunks and burn-in to test persistent
latent state, but that larger replay change is outside this first comparison.

The target branch must use a target RSSM and target Q head. Build online and
target latent features consistently for Double DQN selection and evaluation;
sync the complete target model at the existing target-update cadence. Test
representation drift and Q-target stability before tuning the world-model
loss weights.

## Training schedule and comparison

The pilot uses one seed (27) and one environment at a time. It keeps the task,
partial-observation channel, 600-step episodes, exploration schedule,
evaluation starts, and DQN loss matched to the history-4 GRU-DQN baseline. Its
16-unit deterministic state and two 4-way categorical variables form a compact
latent. To bound runtime, replay updates run every 32 environment steps and
target synchronization every 400 steps. This preserves the baseline's 12.5
optimizer updates per target sync while using one eighth as many DQN updates per
environment step. Record sample count and wall time.

For the first comparison, evaluate these two systems:

1. Existing history-4 GRU-DQN trained end-to-end.
2. Existing history-4 GRU-DQN with its recurrent encoder frozen after 10,000
   episodes.
3. RSSM-DQN with the same Q-learning system and auxiliary prediction losses.
   Save intermediate checkpoints to show the training-duration trajectory.

Do not add imagined rollouts, a policy actor, or an actor-critic in this first
test. Those would change the control algorithm and make it unclear whether a
gain came from the representation or from planning. The pilot trains one
5,000-episode RSSM run with checkpoints at 1,000, 2,500, 3,750, and 5,000
episodes. Epsilon reaches 0.1 at episode 3,750, matching the seed-27 GRU runs.
Compare these with the seed-27 full-GRU 25k policy and frozen-GRU 20k and 25k
policies. The completed result should guide whether a later encoder-freezing
ablation is worth running; the pilot itself does not test that schedule.

Use fixed common evaluation scenarios. Report task return, policy maps over
the grid, observation/reward prediction losses, KL magnitude, and training
wall time. The representation metrics are diagnostics; policy return and wall
time stay the primary outcomes. This is a single-seed feasibility pilot, so
rollout-scenario confidence intervals do not support claims about variation
across training seeds.

## Completed seed-27 pilot result

The pilot completed with one seed and was evaluated on 1,900 shared rollout
scenarios per checkpoint. The best RSSM result was at 3,750 episodes: mean
return 0.3589 in 454.4 seconds of training-loop time. It was below full-GRU
25k (0.4142) and frozen-GRU 20k (0.4542), but above frozen-GRU 25k (0.3275).
It was also below noisy-observation value iteration (0.3989). RSSM return then
fell to 0.1952 at 5,000 episodes despite an increase in its training-return
average. See the [pilot report](RSSM_DQN_PILOT.md) for paired intervals,
timing, configuration, and all policy maps.

The result shows that the compact model can produce a useful policy much
faster in this run, but it does not match the strongest GRU checkpoints. The
RSSM used one eighth the optimizer updates per environment step of the GRU
baseline, so the observed time reduction is not an equal-update architecture
comparison. This feasibility pilot has one RSSM training seed and does not
establish seed-level reliability.

## Main risks

- **More compute:** the RSSM adds latent-state and prediction losses, so
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

