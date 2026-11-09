# RSSM encoder and GRU-DQN comparison

## Result

This is a completed one-seed feasibility comparison (training seed 27) between
the full history-4 GRU-DQN, a GRU frozen after episode 10,000, and a compact
RSSM-inspired latent encoder feeding a DQN. All seven checkpoints were scored
on the same 1,900 rollout scenarios (evaluation seed 17). Intervals below
measure variation over those rollout scenarios; there is only one training
seed in this comparison, so they do not measure seed-to-seed uncertainty.

The best RSSM checkpoint was at 3,750 episodes: mean return 0.3589 after
454.4 seconds of training-loop time (7.57 minutes). It remained below the
full-GRU 25k checkpoint (0.4142) and the frozen-GRU 20k checkpoint (0.4542).
It was 0.0314 above the frozen-GRU 25k checkpoint (0.3275), on paired
scenarios. The RSSM policy also remained 0.0400 below noisy-observation MDP
value iteration (0.3989). Thus the pilot demonstrates a much faster compact
encoder and a useful training-duration curve, but does not show that it
matches the stronger GRU checkpoints.

The RSSM-DQN has 2,979 parameters, compared with 17,608 in the history-4
64-unit GRU-DQN used for the matched seed-27 runs (83.1% fewer parameters).
The last-window training-return column is the mean over the final 1,000
training episodes at each checkpoint.

| Policy checkpoint | Training-loop time | Cumulative seconds per episode | Last-window training return | Mean evaluation return | Paired difference vs noisy-input MDP VI (95% rollout CI) |
|:--|--:|--:|--:|--:|:--|
| RSSM, 1,000 episodes | 85.5 s (1.43 min) | 0.0855 | 0.1198 | 0.1126 | −0.2863 [−0.3073, −0.2652] |
| RSSM, 2,500 episodes | 261.7 s (4.36 min) | 0.1047 | 0.1488 | 0.1726 | −0.2263 [−0.2490, −0.2035] |
| RSSM, 3,750 episodes | 454.4 s (7.57 min) | 0.1212 | 0.1896 | 0.3589 | −0.0400 [−0.0640, −0.0161] |
| RSSM, 5,000 episodes | 671.0 s (11.18 min) | 0.1342 | 0.1716 | 0.1952 | −0.2037 [−0.2275, −0.1799] |
| Full GRU, 25,000 episodes | 3,968.8 s (66.15 min) | 0.1588 | 0.2106 | 0.4142 | +0.0153 [−0.0087, +0.0392] |
| Frozen GRU, 20,000 episodes | 3,102.4 s (51.71 min) | 0.1551 | 0.2604 | 0.4542 | +0.0553 [+0.0318, +0.0787] |
| Frozen GRU, 25,000 episodes | 3,685.4 s (61.42 min) | 0.1474 | 0.2046 | 0.3275 | −0.0714 [−0.0946, −0.0482] |

The RSSM 3,750-episode return was lower than full GRU by 0.0553
(95% rollout CI [−0.0768, −0.0338]) and lower than frozen GRU at 20k by
0.0953 ([−0.1170, −0.0735]). It was higher than frozen GRU at 25k by 0.0314
([+0.0082, +0.0546]). Within the RSSM run, the 3,750 checkpoint beat its
5,000 checkpoint by 0.1637 ([+0.1454, +0.1819]). Training return increased
from 0.1198 in the first 1,000 episodes to 0.1716 in the last 1,000, while
greedy evaluation return fell from the 3,750 checkpoint to 5,000 episodes.
This is evidence that training return alone was not a reliable checkpoint
selection measure in this run.

## Training setup and wall time

- Environment: existing safe-absorbing-states 5×4 task grid; one agent; sensor
  observation error 0.25; action history length 4; 600 steps per episode.
- RSSM seed 27, trained for 5,000 episodes (3,000,000 environment steps).
  Epsilon reached 0.1 at episode 3,750 (decay fraction 0.75). Automaton
  transition actions were selected with probability 0.2.
- AdamW with AMSGrad, learning rate 1e-4, batch size 32, replay capacity
  100,000. RSSM replay optimization ran every 32 environment steps and target
  synchronization every 400 steps. This kept 12.5 optimizer updates per target
  synchronization, as in the GRU configuration, but used one eighth the DQN
  optimizer updates per environment step.
- Model: fused observation/action GRU with 16 deterministic features, two
  categorical latent variables with four classes each, and a 24-feature Q
  input. Total parameter count was 2,979. Auxiliary next-observation and
  reward prediction losses had weights 0.1 each; KL weight was 0.01. The last
  update reported KL 0.00121, observation loss 1.0271, and reward loss
  0.00449.
- The process ran serially from 2026-11-08 18:42:48 UTC to 18:53:59 UTC.
  External process wall time was 673.91 s (11 min 13.91 s); the training loop
  took 670.972 s. User CPU time was 626.77 s and system CPU time 74.74 s.
  Training-loop time per episode averaged 0.1342 s. Checkpoint cumulative
  training-loop times were 85.513 s at 1k, 261.670 s at 2.5k, 454.429 s at
  3.75k, and 670.972 s at 5k.
- The full-GRU 25k process took 3,970.529 s. The frozen-GRU run took
  3,103.937 s to reach 20k and 3,687.345 s to reach 25k. The RSSM's 3.75k
  checkpoint took 88.6% less training-loop time than full-GRU 25k and 85.4%
  less than frozen-GRU 20k. These are observed time-quality points, not
  equal-update architecture benchmarks: the RSSM used fewer updates per
  environment step and had a shorter episode budget.

The seven policy evaluations ran serially from 2026-11-08 19:41:25 UTC to
20:27:05 UTC and took 2,739.970 s (45 min 39.97 s). Each checkpoint used 100
rollouts from each of 19 traversable starts, a 1,000-step horizon, and the same
evaluation seed. The noisy-observation value-iteration baseline averaged
0.398898 on this set. The full-state policy's hidden state was not exposed to
the learned policies.

## Model and interpretation limits

This pilot is an RSSM-inspired recurrent latent encoder, not a canonical full
RSSM. It learns categorical prior and posterior distributions and auxiliary
prediction losses, but does not feed the stochastic latent back into the
deterministic transition. It uses four-token replay windows and resets its
recurrent state at each window. It is trained jointly with the Q head; no
encoder-frozen RSSM stage was tested.

The DQN update cadence differs from the GRU runs. Holding optimizer updates
per target sync constant did not hold updates per environment step constant.
The shorter wall time therefore reflects both the compact 2,979-parameter
model and fewer gradient updates. A follow-up intended to compare encoder
quality at equal data and optimization budgets should match update counts and
use multiple training seeds. No multi-seed RSSM estimate is available here.

An initial set of RSSM implementations was stopped before usable checkpoints
because it was too slow; the final fused small model produced the results
above. The [aborted-attempt log](artifacts/rssm_dqn_pilot/aborted_attempt.json)
preserves those attempts and the [analysis plan](artifacts/rssm_dqn_pilot/analysis_plan.md)
records the pilot intent.

## Visualizations and artifacts

![Policy quality versus training-loop time](artifacts/rssm_dqn_pilot/return_vs_training_time.png)

![Training-return trajectories](artifacts/rssm_dqn_pilot/training_return_trajectories.png)

The [policy-map index](artifacts/rssm_dqn_pilot/policies/index.md) links the
grid-based action and expected-Q visualization for every evaluated checkpoint.
The maps include cold-start and four-repeated-observation panels.
Notebook-style empirical return grids for the RSSM and matched GRU checkpoints
are included in the [research results report](RESEARCH_RESULTS_REPORT.md) and
[grid gallery](artifacts/research_summary/grid_visualizations/index.md).

Machine-readable [raw evaluation rollouts](artifacts/rssm_dqn_pilot/evaluation.json),
[paired analysis](artifacts/rssm_dqn_pilot/analysis.json),
[training manifest](artifacts/rssm_dqn_pilot/training_manifest.json),
[pilot configuration](artifacts/rssm_dqn_pilot/pilot_config.json), and
[training log](artifacts/rssm_dqn_pilot/logs/seed27_rssm_5k.log) preserve the
results and run context. The training checkpoints live under the ignored
`checkpoints_rssm_seed27/` directory.
