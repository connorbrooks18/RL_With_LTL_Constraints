# Frozen-GRU training experiment

## Question

Can the history-4 GRU-DQN keep most of its policy quality while reducing
training wall time by freezing its recurrent encoder partway through training
and updating only its linear Q head afterward?

## Shared protocol

- Runs use paired random seeds and are serial, with arm order alternated by
  seed to limit timing bias from machine load and thermal drift.
- Both arms train for 15,000 episodes × 600 steps, from scratch, with the same
  seed and DQN settings: history length 4, GRU hidden size 64, batch size 32,
  replay capacity 100,000, AdamW learning rate 1e-4, replay optimization
  every four environment steps, target update every 50 steps, and 0.2
  probability of choosing an available automaton transition.
- Epsilon has floor 0.1 and decays over the first 25% of training, reaching
  its floor at episode 3,750 in both arms.
- **End-to-end arm:** train the GRU and Q head jointly for all 15,000
  episodes.
- The frozen arm synchronizes its target GRU at the freeze boundary, freezes
  every online and target GRU parameter, and continues updating only the
  linear Q head. The target Q head continues to update every 50 environment
  steps.
- Within each seed, the two arms should have identical training experience up
  to the freeze point; their exploration trajectories can diverge afterward.
- Evaluate pre-freeze, head-only progress, frozen-final, and end-to-end-final policies on
  identical scenarios: 100 rollouts from each traversable start, with a
  1,000-step horizon. This separates whether head-only learning improves the
  freeze-point policy from whether the final frozen model retains the quality
  of a fully trained GRU. The frozen arm also saves evaluation checkpoints at
  10,000 and 12,500 episodes to show its head-only learning trajectory.

## Decision rule

The primary quality contrast is frozen-final minus end-to-end-final return,
paired by training seed and rollout scenario. An absolute loss of 0.05 in
discounted return is the predeclared non-inferiority margin for this feasibility
study. A positive measured wall-time reduction is the second requirement.
Report paired 95% intervals across training seeds for policy quality and
training time; rollout intervals describe evaluation sampling error only. A
claim that freezing preserves quality and saves time requires the quality
interval's lower bound to exceed −0.05 and the wall-time savings interval's
lower bound to exceed zero. If either interval fails its rule, report the
result as inconclusive rather than treating a favorable mean as proof.

As a secondary, exploratory time-to-quality view, compare the freeze-point,
12,500-episode head-only, and frozen-final checkpoints with the end-to-end
15,000-episode policy. This shows whether head-only training reaches the
registered quality margin earlier and how much trainer-loop time it uses.

## 7,500-episode pilot: seed 17

This pilot trained the frozen arm jointly for 7,500 episodes and then trained
only its Q head for 7,500 more. The end-to-end run took 2,337.452 s; the
frozen run took 2,015.128 s, saving 322.324 s (13.8%). The joint phase took
1,137.611 s and the head-only phase 875.962 s.

On common evaluation scenarios, returns were 0.292102 at the freeze point,
0.431992 at 10k, 0.399685 at 12.5k, and 0.512948 at 15k. The full end-to-end
15k policy returned 0.571598. Thus head-only learning improved the pilot
policy by 0.220846, but the final frozen policy trailed end-to-end training by
0.058650, just outside the predeclared 0.05 margin. The pilot alone does not
establish preserved quality. Its saved checkpoints also verify that the first
7,500 episode returns matched exactly, the GRU tensors remained bitwise
unchanged after freezing, and the Q-head tensors changed.

Because the pilot missed the quality margin slightly, the confirmatory cohort
uses a later freeze point, at episode 10,000. The pilot seed is excluded from
that analysis because it informed the freeze-point choice.

## Confirmatory design: freeze at episode 10,000

- Nine independent paired seeds: 18–26. The seed-17 pilot is excluded from
  confirmatory inference.
- The end-to-end arm trains all 15,000 episodes. The frozen arm trains the
  GRU and head jointly for 10,000 episodes, then trains only the Q head for
  the last 5,000.
- The frozen arm saves checkpoints at 10,000, 12,500, and 15,000 episodes to
  show the head-only learning trajectory.
- Apply the same evaluation scenarios and predeclared decision rule above.
- The integrity analysis checks that online GRU tensors stay bitwise fixed,
  the target GRU remains synced to those frozen tensors, and Q-head tensors
  change at head-only checkpoints.

## Confirmatory results and artifacts

All 18 training runs (nine paired seeds, 18–26) completed serially. The exact
commands, episode-level returns, checkpoint paths, timestamps, and per-run wall
times are retained in the [training manifest](artifacts/gru_freeze_ab/confirmatory_10k/training_manifest.json)
and individual logs under `artifacts/gru_freeze_ab/confirmatory_10k/logs/`.
Training checkpoints are in the ignored `checkpoints_gru_freeze_ab/` directory.

The final evaluation covered all 36 checkpoints: each seed's end-to-end 15k
policy and its frozen-arm checkpoints at 10k, 12.5k, and 15k. Every checkpoint
used the same 1,900 rollout scenarios (100 rollouts from each of 19 traversable
starts, 1,000-step horizon, evaluation seed 17). The batch ran from
2026-11-06 12:18:58 UTC to 13:56:49 UTC and took 5,871.590 seconds (1 h 37 min
52 s). See the [raw evaluation](artifacts/gru_freeze_ab/confirmatory_10k/evaluation.json)
and [paired analysis](artifacts/gru_freeze_ab/confirmatory_10k/analysis.json).

| Seed | Full process s | Frozen process s | Time saved s (%) | Frozen final return | Full final return | Frozen − full |
|---:|---:|---:|---:|---:|---:|---:|
| 18 | 2,334.009 | 2,127.542 | 206.466 (8.85%) | 0.474 | 0.435 | +0.039 |
| 19 | 2,328.561 | 2,124.016 | 204.545 (8.78%) | 0.518 | 0.461 | +0.057 |
| 20 | 2,335.165 | 2,119.153 | 216.012 (9.25%) | 0.373 | 0.461 | −0.088 |
| 21 | 2,330.316 | 2,114.155 | 216.162 (9.28%) | 0.446 | 0.676 | −0.229 |
| 22 | 2,331.480 | 2,119.765 | 211.715 (9.08%) | 0.433 | 0.607 | −0.174 |
| 23 | 2,331.709 | 2,120.486 | 211.223 (9.06%) | 0.591 | 0.262 | +0.329 |
| 24 | 2,332.798 | 2,134.793 | 198.005 (8.49%) | 0.481 | 0.511 | −0.031 |
| 25 | 2,326.675 | 2,117.447 | 209.228 (8.99%) | 0.554 | 0.645 | −0.091 |
| 26 | 2,337.102 | 2,122.082 | 215.021 (9.20%) | 0.636 | 0.504 | +0.132 |
| **Mean** | **2,331.979** | **2,122.160** | **209.820 (9.00%)** | **0.501** | **0.507** | **−0.006** |

Across the nine independent training seeds, the frozen-minus-full return
contrast was −0.006 (95% Student-t CI [−0.136, +0.124]). The registered
non-inferiority margin is 0.05, so the lower confidence bound does not exceed
−0.05. **The experiment does not establish that freezing preserves policy
quality.** The result is inconclusive for quality, despite the near-zero mean
contrast; seed-level variation is substantial.

The process wall-time saving was 209.820 s per pair (95% CI [205.190,
214.449]), or 9.00% (95% CI [8.80%, 9.19%]). This passes the registered
positive-saving criterion. The joint training loop took about 10,000 episodes
and the head-only phase 5,000 episodes. At the intermediate 12.5k checkpoint,
mean return was 0.512 (95% CI [0.459, 0.565]); its paired change from the
10k freeze point was +0.015 (95% CI [−0.028, +0.058]). The secondary
checkpoint comparisons do not meet the registered quality-margin rule either.

All integrity checks passed for all nine seeds: training episode returns
matched between arms through the 10,000-episode freeze boundary, online and
target GRU tensors remained bitwise unchanged afterward, and Q-head tensors
changed at the 12,500- and 15,000-episode checkpoints.

Policy maps for all 36 checkpoints, including action and expected-Q grids, are
linked from the [policy-map index](artifacts/gru_freeze_ab/confirmatory_10k/policies/index.md).
The paired summary figure is [comparison.png](artifacts/gru_freeze_ab/confirmatory_10k/comparison.png).

## Direct-stop demonstration: stop at 12,500 episodes

To measure an actual early stop without adding a large seed cohort, one
additional serial matched pair (seed 27) compared the full 15,000-episode run
with a run that trained the GRU and Q head jointly through episode 10,000,
then froze the GRU and stopped after 2,500 Q-head-only episodes. The frozen
arm stopped at episode 12,500. Its epsilon schedule used a 0.30 decay fraction
so the exploration floor was reached at the same episode (3,750) as in the
15,000-episode control's 0.25 schedule. Both used history length 4, a 600-step
episode horizon, GRU size 64, batch size 32, replay capacity 100,000, learning
rate 1e-4, optimization every 4 environment steps, and target copies every 50
steps. The frozen phase left 520 Q-head parameters trainable.

| Arm | Episodes | Start UTC | Finish UTC | Process wall time | Evaluation mean return |
|:---|---:|:---|:---|---:|---:|
| Full GRU | 15,000 | 2026-11-06 14:06:04 | 2026-11-06 14:46:44 | 2,440.043 s | 0.460117 |
| GRU frozen after 10k | 12,500 | 2026-11-06 15:29:55 | 2026-11-06 16:00:41 | 1,845.933 s | 0.557321 |
| **Difference** | **2,500 fewer** |  |  | **594.110 s (24.35%) saved** | **+0.097204 frozen − full** |

The full run's training loop took 2,438.296 s. In the frozen run, the first
10,000 joint-training episodes took 1,549.717 s and the final 2,500 head-only
episodes took 294.568 s (1,844.285 s total training-loop time). Remaining
process time covers startup and checkpoint handling. Per-episode process time
was 0.16267 s full and 0.14767 s frozen. Both schedules used 600 environment
steps per episode.

Evaluation used 1,900 common rollouts per policy (100 per traversable start,
1,000-step horizon, evaluation seed 17) and took 500.808 s for the three
checkpoints. The paired frozen-minus-full return difference was +0.097204;
the 95% rollout-scenario interval was [+0.077363, +0.117045]. The frozen
12.5k policy also returned 0.158423 more than noisy-input value iteration
(95% rollout interval [+0.133265, +0.183580]). These intervals describe
scenario sampling for this one trained seed only. They do not estimate
seed-to-seed variation and do not establish that the quality result will
repeat across training runs. The nine-seed cohort above remains inconclusive
for quality preservation.

Integrity checks passed: episode returns matched through the 10,000-episode
freeze point, online GRU weights were bitwise unchanged after freezing, the
target GRU matched the frozen online GRU, and the Q head changed. The direct
run's checkpoint maps are in the [policy-map index](artifacts/gru_freeze_ab/direct_stop_12500_matched/policies/index.md);
machine-readable [timing and paired analysis](artifacts/gru_freeze_ab/direct_stop_12500_matched/direct_stop_analysis.json),
[evaluation rollouts](artifacts/gru_freeze_ab/direct_stop_12500_matched/evaluation.json),
[training manifest](artifacts/gru_freeze_ab/direct_stop_12500_matched/training_manifest.json),
and [training logs](artifacts/gru_freeze_ab/direct_stop_12500_matched/logs/) preserve the run details.

This one-seed direct-stop run demonstrates a concrete wall-time reduction
while producing a competitive policy in this evaluation. It is a focused
demonstration, not confirmatory evidence of non-inferiority. No additional
multi-seed expansion of this direct-stop protocol was run. An initial attempt
used the same fractional epsilon-decay setting for both episode budgets, so
the shorter run reached the exploration floor earlier and its experience prefix did not match. That
attempt is excluded; its diagnostic files remain under
`artifacts/gru_freeze_ab/direct_stop_12500/`. The reported run adjusts the
shorter arm's decay fraction to preserve the shared episode-3,750 floor.

## One-seed 25,000-episode head-only trajectory

The nine-seed cohort ended at 15,000 episodes, so one serial paired extension
(seed 27) ran both arms to 25,000 episodes. Both schedules reached epsilon
0.1 at episode 3,750 (`epsilon_decay_fraction=0.15`). The GRU froze after
10,000 episodes in the treatment arm; only its 520-parameter Q head changed
after that point. Other settings matched the prior runs: history 4, hidden
size 64, 600 steps per episode, batch 32, replay 100,000, AdamW 1e-4,
optimization every 4 environment steps, and target copies every 50 steps.
Both arms shared an exact episode-return prefix through the freeze boundary.

The full-GRU run started at 2026-11-06 16:13:47 UTC and finished at 17:19:57;
its process wall time was 3,970.529 s (training loop 3,968.804 s). The frozen
run followed serially from 17:19:57 to 18:21:25; process wall time was
3,687.345 s (training loop 3,685.384 s), including 1,937.916 s of joint
training and 1,747.468 s of head-only training. At the 20k checkpoint, elapsed
process time was 3,103.937 s, saving 866.593 s (21.83%) versus the full 25k
process. At equal 25k budgets, freezing saved 283.184 s (7.13%).

Five checkpoints were evaluated serially on 1,900 common scenarios each (100
rollouts from each of 19 traversable starts, 1,000-step horizon, evaluation
seed 17); the batch took 825.296 s. The frozen policy's mean returns were
0.510470 at 10k, 0.499961 at 15k, 0.454157 at 20k, and 0.327506 at 25k.
The full-GRU 25k return was 0.414177. The paired frozen-20k minus full-25k
difference was +0.039980 (95% rollout-scenario CI [+0.025187, +0.054772]),
which clears the −0.05 non-inferiority margin for this one seed. The 20k
checkpoint was reached in 3,103.937 s, saving 21.83% wall time with a
competitive policy in this run.

The longer head-only run did **not** converge to a stable policy under the
predeclared plateau criterion. From 20k to 25k, return fell by 0.126651 (95%
rollout-scenario CI [−0.144735, −0.108568]); the interval is well outside the
±0.02 practical-stability band. At 25k, the frozen policy was 0.086671 below
the full-GRU policy (95% rollout-scenario CI [−0.107068, −0.066275]). Thus,
the evidence supports stopping at a competitive earlier checkpoint in this
run; it does not support continuing head-only training indefinitely or claim
that the 25k head-only policy converged. These intervals quantify rollout
scenario variation for seed 27, not training-seed variation. The existing
nine-seed study remains inconclusive for across-seed policy-quality
preservation.

All integrity checks passed: returns matched through 10k, online and target
GRU weights stayed bitwise fixed at 10k, 15k, 20k, and 25k, and the Q head
changed after freezing. The [predeclared convergence analysis plan](artifacts/gru_freeze_ab/convergence_25k/analysis_plan.md),
[machine-readable analysis](artifacts/gru_freeze_ab/convergence_25k/analysis.json),
[paired rollout data](artifacts/gru_freeze_ab/convergence_25k/evaluation.json),
[training manifest](artifacts/gru_freeze_ab/convergence_25k/training_manifest.json),
[training logs](artifacts/gru_freeze_ab/convergence_25k/logs/),
[return trajectory figure](artifacts/gru_freeze_ab/convergence_25k/convergence.png),
and [eight policy maps](artifacts/gru_freeze_ab/convergence_25k/policies/index.md)
retain the full result.

Reproduce training with:

```sh
venv/bin/python run_gru_freeze_comparison.py
```

Regenerate the policy maps with:

```sh
venv/bin/python plot_gru_freeze_policies.py \
  artifacts/gru_freeze_ab/confirmatory_10k/training_manifest.json \
  --output-dir artifacts/gru_freeze_ab/confirmatory_10k/policies
```

The one-seed pilot evaluation is in
`artifacts/gru_freeze_ab/pilot_evaluation_seed17.json`.
