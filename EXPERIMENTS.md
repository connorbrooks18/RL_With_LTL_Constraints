# Experiment protocol

## Research question

Can a history-4 GRU-DQN learn temporal-logic motion planning with partial grid
observations better than a feedforward DQN that sees only the latest reading?
Can freezing the GRU or using an RSSM-trained encoder reduce training time
while retaining useful policy quality?

Experiment 1 has the core 5×4 task and the Office Task 1 transfer.
Experiment 2 compares full-GRU training, GRU freezing, and an RSSM encoder.
The result report uses one representative median outcome for each comparison.

## Shared observation contract

Every learned controller receives the current automaton state `q` and Rabin-
pair index. The grid coordinate is noisy. The GRU receives a four-observation
history and the preceding actions; the MLP receives only the latest reading.
Both use the same `q`-derived legal-action mask.

## Experiment 1A: core grid comparison

- Environment: one robot in the repository's 5×4 task,
  `(F G a | F G b) & G !c`.
- Grid observation error: 0.25.
- Longest completed budget: 60,000 episodes × 600 steps.
- GRU: one 64-unit recurrent layer. MLP: two 64-unit hidden layers.
- Batch size 32; learning rate `1e-4`; replay capacity 100,000.
- Replay update every four environment steps; target update every 50 steps.
- Epsilon decreases from 1.0 to 0.1 over the first quarter of training.
- Automaton-action probability 0.2; discount 0.99999 outside accepting
  states and 0.99 inside accepting states.
- Evaluation: 100 paired rollouts from each traversable start, with a
  1,000-step horizon.

For a concise view, the report selects the median paired return difference
from the longest completed comparison. This is a representative policy pair,
not an estimate of between-training-run uncertainty.

## Experiment 1B: Office Task 1 transfer

The Office task from Li's thesis, §4.5.2, Task 1, is

\[
\varphi_{office}=\Box\Diamond( Print\land\Diamond(a\lor c))\land\Box\neg S.
\]

The robot repeatedly visits the printer to collect documents, delivers them
to Office `a` or `c`, and never enters storage `S`. The local proposition
mapping is `p = Print` and `s = S`. The intended observation is the complete
north/south/east/west wall, hallway, door, and window signature. Rooms with
the same signature share the observation probability analytically.

Keep the grounded acceptance reward used by the other experiments:
`1 − 0.99 = 0.01`, with discount 0.99999 outside accepting states and 0.99
inside them. The DQN target clamp to `[0,1]` remains consistent with that
reward scale. Keep `q` visible and use the same initial-state distribution
for training and evaluation.

The transfer uses 60,000 episodes × 600 steps per episode for both the
history-4 GRU and current-reading MLP. The longer horizon increases the
chance of reaching the printer and delivering documents within one episode
while preserving the existing objective and reward. Track completed
printer-to-office cycles and storage violations in addition to return. The
current training run is serial. Validate the transition and analytic
observation tables against Junchao's SARSOP model before interpreting policy
learning. The reference policy is an explicit-model benchmark, not a
neural-network training result.

References: [Li's thesis](https://doi.org/10.25820/etd.007512), §4.5.2,
Task 1, equations (47)–(48); [Junchao's full-observation SARSOP Office model
and outputs](https://github.com/JunchaoLi001/LDGBA-Model_Checking/tree/main/Office%20Simulation/Office%20full%20obsv%20by%20SARSOP).

## Experiment 2: reduce training time

All methods use the q-visible, partially observed core task and a
60,000-episode × 600-step budget, matching Experiment 1A. The frozen arm
trains the GRU and Q head together until episode 10,000, then trains only the
Q head. The RSSM arm learns a compact latent encoder jointly with a DQN head.
The comparison uses one fixed training run per method, executes one at a
time, and reports only final endpoints.

| Method | Encoder training | Final evaluation |
|:--|:--|:--|
| Full GRU | Train throughout | Mean return and trainer-reported elapsed time |
| Frozen GRU | Train jointly through episode 10,000, then freeze | Same evaluation |
| RSSM encoder | Train latent encoder with DQN | Same evaluation |

This single-run comparison is descriptive and cannot measure variation
between training runs. Use a fixed task-success target and report wall time,
successful delivery cycles, and storage violations alongside return.
