# Automaton-state input ablation

## Question

The earlier MLP-DQN did **not** receive the full product observation. It saw
the Rabin-pair index `i` and noisy row and column only. This task has one
Rabin pair, so `i` is always 0; it does not encode automaton progress. The
history-4 GRU also does not receive the true automaton state `q`: it sees the
same observations,
the preceding actions, and a persistent record of its own selected epsilon
target. That epsilon-target feature records one controller decision; it is not
the automaton's current state, which can also change as labels are consumed.

This seed-17 follow-up asks whether providing the MLP with the true `q` closes
its performance gap, and compares it with exact value-iteration policies that
receive inferred or true `q` at decision time.

## Controlled comparison

The primary MLP arm receives `(i, q, noisy_row, noisy_column)` but keeps the
same global action interface as the partial-observation GRU and MLP. Every
grid move and every automaton edge is available; an unavailable epsilon edge
acts as a no-op. This isolates adding `q` to the input without also revealing
the true state's legal-action mask. The two MLPs have the same two 64-unit
hidden layers and the same training schedule; the q-observed MLP has 14 input
features and 5,640 parameters, versus 10 features and 5,384 parameters for the
no-q MLP. The GRU has 17,608 parameters.

| Setting | q-observed MLP |
|:--|:--|
| Training seed | 17 |
| Episodes × horizon | 15,000 × 600 (9,000,000 environment steps) |
| Optimizer / learning rate | AdamW / 1e-4 |
| Replay / batch | 100,000 / 32 |
| Replay optimization | Every 4 environment steps |
| Target-network copy | Every 50 environment steps |
| Epsilon schedule | 1.0 to 0.1 over the first 25% of episodes, then 0.1 |
| Automaton-edge exploration probability | 0.2 |
| Observation error | 0.25 on physical position; true q is observed |
| Discounts | 0.99999 nonaccepting; 0.99 accepting |
| Training wall time | 1,820.729 s (30 min 20.729 s), serial |
| Training timestamps (UTC) | 2026-11-09 20:31:28.188 to 21:01:48.937 |
| Training return | Mean 0.11028; first 1k 0.10516; last 1k 0.10650 |

Evaluation used 100 rollouts from each of the 19 traversable starting squares,
a 1,000-step horizon, and common movement/observation seeds (1,900 matched
scenarios per policy). The evaluation ran from 2026-11-09 21:01:59.249 to
21:10:58.067 UTC and took 538.833 s (8 min 58.833 s). The intervals below are
95% rollout-scenario intervals; they do not estimate variation across training
seeds. Value iteration used threshold `1e-8`.

## Results

| Policy | What it observes | Mean discounted return (95% rollout CI) |
|:--|:--|--:|
| History-4 GRU-DQN, seed 17 | Recent noisy positions, actions, remembered epsilon target; true q hidden | 0.57160 [0.54944, 0.59375] |
| Current-observation MLP-DQN, seed 17 | Noisy position; true q hidden | 0.10526 [0.09146, 0.11906] |
| q-observed MLP-DQN, seed 17 | True q and noisy position | 0.10368 [0.08997, 0.11739] |
| Exact value iteration, inferred q | Noisy position and its own inferred q | 0.39890 [0.37688, 0.42092] |
| Exact value iteration, true q | True q and noisy position | 0.53306 [0.51063, 0.55550] |
| Full-state VI oracle | True q and true position | 0.76462 [0.74562, 0.78362] |

The q-observed MLP did not improve over its no-q counterpart: the paired
difference was −0.00158 (95% rollout CI [−0.02097, +0.01780]). The GRU beat
the q-observed MLP by +0.46792 ([+0.44548, +0.49036]). The GRU also beat the
true-q/noisy-position value-iteration policy by +0.03853 ([+0.01546,
+0.06160]) on these paired rollouts. Both learned and VI comparisons use the
same scenarios, but the q-aware VI policy is given true q while the GRU is
not.

These results separate representational access from successful learning. The
exact q-aware VI policy scores 0.533, showing that true q plus the noisy
position channel supports a strong policy. The q-observed DQN scores only
0.104 and its training return stays nearly flat around 0.11, so this run did
not learn that policy. The GRU's advantage over q-aware VI is much smaller
than its advantage over the q-aware MLP; that pattern is consistent with the
MLP's optimization/training result being the main weakness here. It does not
prove that recurrence is intrinsically better than a sufficiently trained
feedforward policy.

### Return grids by starting square

Each table uses the same 5×4 arrangement as the task. Values are mean
discounted rollout returns; `B` marks the obstacle. These are empirical
start-state returns, not learned scalar value estimates.

**History-4 GRU-DQN**

| Row \ Col | 0 | 1 | 2 | 3 |
|:--:|--:|--:|--:|--:|
| 0 | 0.464 | 0.537 | 0.000 | 0.870 |
| 1 | 0.480 | 0.489 | 0.700 | 1.000 |
| 2 | B | 0.422 | 0.000 | 0.810 |
| 3 | 1.000 | 0.670 | 1.000 | 0.840 |
| 4 | 0.890 | 0.000 | 0.690 | 0.000 |

**Current-observation MLP-DQN (q hidden)**

| Row \ Col | 0 | 1 | 2 | 3 |
|:--:|--:|--:|--:|--:|
| 0 | 0.000 | 0.000 | 0.000 | 0.000 |
| 1 | 0.000 | 0.000 | 1.000 | 0.000 |
| 2 | B | 0.000 | 0.000 | 0.000 |
| 3 | 0.000 | 0.000 | 1.000 | 0.000 |
| 4 | 0.000 | 0.000 | 0.000 | 0.000 |

**q-observed MLP-DQN**

| Row \ Col | 0 | 1 | 2 | 3 |
|:--:|--:|--:|--:|--:|
| 0 | 0.000 | 0.010 | 0.000 | 0.000 |
| 1 | 0.000 | 0.010 | 0.080 | 1.000 |
| 2 | B | 0.000 | 0.000 | 0.080 |
| 3 | 0.580 | 0.010 | 0.140 | 0.000 |
| 4 | 0.060 | 0.000 | 0.000 | 0.000 |

**Exact VI with inferred q and noisy position**

| Row \ Col | 0 | 1 | 2 | 3 |
|:--:|--:|--:|--:|--:|
| 0 | 0.470 | 0.330 | 0.000 | 0.130 |
| 1 | 0.360 | 0.370 | 0.600 | 1.000 |
| 2 | B | 0.350 | 0.000 | 0.440 |
| 3 | 1.000 | 0.300 | 0.930 | 0.470 |
| 4 | 0.260 | 0.000 | 0.570 | 0.000 |

**Exact VI with true q and noisy position**

| Row \ Col | 0 | 1 | 2 | 3 |
|:--:|--:|--:|--:|--:|
| 0 | 0.600 | 0.590 | 0.000 | 0.290 |
| 1 | 0.630 | 0.540 | 0.640 | 1.000 |
| 2 | B | 0.620 | 0.000 | 0.640 |
| 3 | 1.000 | 0.500 | 1.000 | 0.710 |
| 4 | 0.630 | 0.000 | 0.740 | 0.000 |

**Full-state VI oracle (true q and true position)**

| Row \ Col | 0 | 1 | 2 | 3 |
|:--:|--:|--:|--:|--:|
| 0 | 0.997 | 0.993 | 0.000 | 1.000 |
| 1 | 0.985 | 0.993 | 0.800 | 1.000 |
| 2 | B | 0.983 | 0.000 | 1.000 |
| 3 | 1.000 | 0.997 | 1.000 | 1.000 |
| 4 | 1.000 | 0.000 | 0.780 | 0.000 |

## Figures and data

The return maps retain the notebook `GridMDP.plot` rendering style. Difference
maps use the same paired rollouts. Action maps average Q values over the
position-noise distribution; each arrow is the most common greedy action and
its number gives that action's frequency.

![History-4 GRU start returns](artifacts/pomdp_automaton_state_ablation/gru_returns_by_start.png)

![q-observed MLP start returns](artifacts/pomdp_automaton_state_ablation/mlp_q_observed_returns_by_start.png)

![GRU minus q-observed MLP](artifacts/pomdp_automaton_state_ablation/gru_minus_q_mlp_by_start.png)

![GRU minus true-q value iteration](artifacts/pomdp_automaton_state_ablation/gru_minus_q_vi_by_start.png)

![History-4 GRU action and Q map](artifacts/pomdp_automaton_state_ablation/history4_gru_policy.png)

![q-observed MLP action and Q map](artifacts/pomdp_automaton_state_ablation/q_observed_mlp_policy.png)

The [complete grid gallery](artifacts/pomdp_automaton_state_ablation/index.md)
contains all ten return and paired-difference maps. The raw paired rollout data
are in [evaluation_q_observed.json](artifacts/pomdp_automaton_state_ablation/evaluation_q_observed.json).
The q-observed checkpoint and sidecar are in
`checkpoints_pomdp_mlp_q_observed_seed17/`.

## Initial q-observed MLP diagnostic with a privileged action mask

Before controlling the action interface, a separate seed-17 MLP was trained
with true q, noisy physical position, and the true state's legal-action mask.
It did **not** receive the full product observation because physical position
remained noisy. It scored 0.21210 (95% rollout CI [0.19371, 0.23048]),
compared with 0.10526 for the no-q MLP. That result changed both the input and
the action mask, so it cannot identify the effect of q alone. Under the controlled global-action
interface, adding q did not help this seed. The original run remains available
in [evaluation.json](artifacts/pomdp_automaton_state_ablation/evaluation.json)
and its checkpoint is `checkpoints_pomdp_mlp_product_state_seed17/`.

## Scope

This is one matched training seed. Scenario confidence intervals quantify
rollout sampling only; they do not replace multiple independently trained
seeds. The true-q VI policy is an exact-MDP reference with privileged q, not a
POMDP solver. The full-state VI policy also receives the true physical
position and is an oracle bound for context.
