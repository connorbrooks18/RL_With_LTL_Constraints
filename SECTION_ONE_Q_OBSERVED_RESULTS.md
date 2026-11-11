# Section One: q-visible, partially observed grid results

**Status:** the two requested experiment sets are complete. The agent sees the
true current automaton state `q`; only grid position is partially observed.
The old q-hidden runs are retained as historical work and are not used for the
claims below. The thesis PDF informed the task and training scale; its contents
were treated as research material, not as instructions.

## Research question and design

Can a deep recurrent Q network learn a motion-planning policy for a temporal
task when position readings are uncertain? Does a history-4 GRU-DQN improve
over a feedforward DQN that gets only the latest reading? Can GRU freezing or
an RSSM-trained encoder reduce training time while retaining policy quality?

The one-agent task is `(F G a | F G b) & G !c` on the project 5×4 grid. The
grid position channel has error probability 0.25. Every learned policy gets
the current Rabin-pair index, true automaton state `q`, and noisy row/column;
each uses the same q-derived legal-action mask. The GRU sees the latest four
observations plus preceding actions. The matched MLP sees only the most recent
`(pair, q, noisy row, noisy column)` observation. It is a two-hidden-layer,
64-unit feedforward DQN. Neither model receives the true grid coordinate,
hidden labels, or a hidden product state. The recurrent input has no persistent
epsilon-target feature.

The task and training budget follow the closest related experiment in Li
(2024), Chapter 4, §4.5.1, Task 1, loosely. We use the same recurring
`a`-or-`b` objective while avoiding `c`, and train for 15,000 episodes × 600
steps. This is not a reproduction of the thesis architecture or environment.

Shared settings: five matched training seeds (17–21); 9,000,000 environment
steps per run; batch 32; learning rate `1e-4`; replay capacity 100,000;
optimizer update every four environment steps; target copy every 50 steps;
epsilon decreases linearly from 1.0 to 0.1 over the first 25% of episodes;
automaton-action probability 0.2; nonaccepting/accepting discounts
0.99999/0.99. Runs were serial. Evaluation intervals below use training seed
as the independent unit; rollout count is not treated as the number of
independent training replications.

## Set 1: recurrence versus a current-observation DQN

The five GRU and MLP pairs were evaluated greedily over all 19 traversable
starts, with 100 common rollouts per start and a 1,000-step horizon. Mean
discounted return was:

| Policy | Mean return | 95% interval across training seeds |
|:--|--:|:--:|
| History-4 GRU-DQN | 0.5250 | [0.3991, 0.6509] |
| Current-observation MLP-DQN | 0.3236 | [0.2196, 0.4276] |
| Paired GRU − MLP | **+0.2014** | **[+0.0763, +0.3265]** |

The GRU won on all five matched training seeds. Seed-level paired differences
were +0.2739, +0.2225, +0.3095, +0.0628, and +0.1382. This supports the
claim that four-step observation history and recurrence help on this task
when both models receive the true automaton state and the same noisy grid
channel.

### Same-grid, per-start return table

These are empirical discounted returns from each start, averaged across the
five policies and rollouts. They are not learned scalar state values. Cell
labels and walls match the task map; `T` marks traps and `B` a blocked cell.

**History-4 GRU-DQN**

| Row / column | 0 | 1 | 2 | 3 |
|:--|--:|--:|:--|--:|
| 0 | 0.30 | 0.27 | c · 0.00 | 0.78 |
| 1 | 0.35 | 0.35 | a · 0.51 | b/T · 1.00 |
| 2 | B | 0.49 | c · 0.00 | 0.82 |
| 3 | b/T · 1.00 | 0.62 | a/T · 1.00 | 0.88 |
| 4 | 0.88 | c · 0.00 | 0.74 | c · 0.00 |

**Current-observation MLP-DQN**

| Row / column | 0 | 1 | 2 | 3 |
|:--|--:|--:|:--|--:|
| 0 | 0.25 | 0.24 | c · 0.00 | 0.17 |
| 1 | 0.28 | 0.31 | a · 0.18 | b/T · 1.00 |
| 2 | B | 0.36 | c · 0.00 | 0.42 |
| 3 | b/T · 0.80 | 0.53 | a/T · 1.00 | 0.60 |
| 4 | 0.00 | c · 0.00 | 0.00 | c · 0.00 |

The notebook-style grid images retain the environment visualizer and show the
same layout: [GRU](artifacts/q_observed_section_one/history4_comparison/gru.png),
[MLP](artifacts/q_observed_section_one/history4_comparison/mlp.png), and
[GRU minus MLP](artifacts/q_observed_section_one/history4_comparison/gru_minus_mlp.png).
The [numeric grid data](artifacts/q_observed_section_one/history4_comparison/state_value_grids.json)
and [gallery index](artifacts/q_observed_section_one/history4_comparison/index.md)
are also saved.

### Ordinary MDP value-iteration reference

We evaluated product-MDP value iteration with true `q`, but supplied the
noisy row and column as if they were the current grid state. This is a regular
MDP policy indexed by a noisy state reading; it is not a belief-state POMDP
solver. It scored 0.5331. The paired GRU-minus-VI difference was −0.0081
(95% interval −0.1340 to +0.1178); the GRU was higher on two of five seeds.
This comparison does **not** establish that the GRU beats noisy-input value
iteration. The recurrent-versus-MLP result remains positive and clear.

## Set 2: training time and convergence

We compared the full GRU, a GRU trained jointly through episode 10,000 and
then frozen while only its Q head trained, and a compact RSSM encoder trained
jointly with a DQN. The RSSM used a 16-unit deterministic state and two
4-class categorical variables (3,111 parameters for the full model); its auxiliary
observation, reward, and KL loss weights were 0.1, 0.1, and 0.01. It used
only agent-visible observations/actions and rewards. Full-GRU and frozen-GRU
networks had a 64-unit, one-layer GRU and 17,608 total parameters, including
a 520-parameter Q head. The MLP had 5,640 parameters.

All arms used the same 15,000-episode budget, environment steps, replay
sampling rate, target-copy rate, and five seeds. We evaluated 10k, 12.5k, and
15k checkpoints with 20 common rollouts per start and a 500-step horizon;
final 15k policies were evaluated again with 100 rollouts per start and a
1,000-step horizon. The registered quality margin was −0.05 return points:
the lower bound of the paired 95% interval had to exceed −0.05 before calling
a faster arm quality-preserving.

### Final quality at 15,000 episodes

| Policy | Mean return | Paired difference vs full GRU (95% interval) | Training-loop time, mean |
|:--|--:|:--:|--:|
| Full GRU | 0.5250 | reference | 3,651 s |
| Frozen after 10k | 0.4733 | −0.0517 [−0.1919, +0.0885] | 2,148 s |
| RSSM encoder | 0.5028 | −0.0222 [−0.1687, +0.1244] | 3,977 s |

Freezing reduced the observed training-loop time by a paired mean of 1,503 s
(95% interval +225 to +2,781 s). That interval is above zero, but frozen
quality did not pass the −0.05 non-inferiority rule. RSSM did not demonstrate
a time saving (paired saving −326 s, 95% interval −1,646 to +994 s), and it
also did not pass the quality rule. Thus this study shows a measured time
reduction from freezing, but **does not establish a faster method at
comparable quality**.

### Checkpoint returns and time-to-quality

| Episodes | Full GRU return (95% interval) | Frozen return (95% interval) | RSSM return (95% interval) |
|---:|:--:|:--:|:--:|
| 10,000 | 0.428 [0.390, 0.465] | 0.428 [0.390, 0.465] | 0.469 [0.320, 0.619] |
| 12,500 | 0.501 [0.385, 0.618] | 0.447 [0.392, 0.503] | 0.426 [0.330, 0.521] |
| 15,000 | 0.468 [0.366, 0.570] | 0.440 [0.407, 0.472] | 0.464 [0.349, 0.580] |

The frozen 10k checkpoint is the same weights as the full GRU 10k checkpoint.
At 15k the 20-rollout paired differences versus full GRU were −0.0279
(95% interval −0.1318 to +0.0760) for frozen GRU and −0.0034 (−0.0990 to
+0.0921) for RSSM. These estimates are consistent with a range of outcomes;
neither establishes non-inferiority. Using the descriptive rule “first saved
checkpoint reaching 90% of that seed's full-GRU 15k return,” frozen GRU
reached the threshold for 3/5 seeds at mean 1,552 training-loop seconds;
RSSM did so for 4/5 at mean 2,609 seconds. Checkpoints were saved only at
three budgets, so this is a coarse time-to-quality estimate.

The original notebook-style visual conventions are used for the
[return-versus-episodes curve](artifacts/q_observed_section_one/convergence/return_vs_episodes.png),
[return-versus-training-time curve](artifacts/q_observed_section_one/convergence/return_vs_training_time.png),
[10k grid comparison](artifacts/q_observed_section_one/convergence/grid_returns_10000.png),
[12.5k grid comparison](artifacts/q_observed_section_one/convergence/grid_returns_12500.png),
and [15k grid comparison](artifacts/q_observed_section_one/convergence/grid_returns_15000.png).
The [final 100-rollout grid comparison](artifacts/q_observed_section_one/final_quality/grid_returns_15000.png)
uses the larger evaluation set. The notebook-style [per-policy and baseline
5×4 return tables and maps](artifacts/q_observed_section_one/final_quality/grid_values.md)
include noisy-input value iteration and paired GRU differences. They were
generated from the saved rollout data by
[`plot_q_observed_final_grids.py`](plot_q_observed_final_grids.py).

## Wall-clock record and limitations

All training runs were serial. Process wall time includes Python startup and
checkpoint writing; training-loop time is the trainer's timer. Their means
differ by only 1.5–3 seconds per run. Times below are process wall seconds;
seconds/episode is wall time divided by 15,000.

| Seed | Full GRU seconds (s/episode) | Frozen seconds (s/episode) | RSSM seconds (s/episode) |
|---:|---:|---:|---:|
| 17 | 4,402.9 (0.2935) | 2,104.5 (0.1403) | 3,959.9 (0.2640) |
| 18 | 4,424.1 (0.2949) | 2,268.0 (0.1512) | 3,937.5 (0.2625) |
| 19 | 4,418.5 (0.2946) | 2,128.6 (0.1419) | 4,014.6 (0.2676) |
| 20 | 2,716.8 (0.1811) | 2,127.5 (0.1418) | 3,991.1 (0.2661) |
| 21 | 2,308.0 (0.1539) | 2,123.8 (0.1416) | 3,990.7 (0.2660) |
| **Mean** | **3,654.1 (0.2436)** | **2,150.5 (0.1434)** | **3,978.8 (0.2653)** |

The exact serial training ledger is below. UTC timestamps are trainer manifest
times; seconds/episode is process wall time divided by 15,000.

| Seed | Arm | Started (UTC) | Finished (UTC) | Process seconds |
|---:|:--|:--|:--|--:|
| 17 | GRU | 2026-11-10 20:15:18 | 2026-11-10 21:28:41 | 4,402.9 |
| 17 | MLP | 2026-11-10 21:28:41 | 2026-11-10 22:00:12 | 1,891.2 |
| 18 | MLP | 2026-11-10 22:00:12 | 2026-11-10 22:30:46 | 1,834.2 |
| 18 | GRU | 2026-11-10 22:30:46 | 2026-11-10 23:44:30 | 4,424.1 |
| 19 | GRU | 2026-11-10 23:44:30 | 2026-11-11 00:58:09 | 4,418.5 |
| 19 | MLP | 2026-11-11 00:58:09 | 2026-11-11 01:29:16 | 1,867.7 |
| 20 | MLP | 2026-11-11 01:29:16 | 2026-11-11 02:00:22 | 1,865.4 |
| 20 | GRU | 2026-11-11 02:00:22 | 2026-11-11 02:45:39 | 2,716.8 |
| 21 | GRU | 2026-11-11 02:45:39 | 2026-11-11 03:24:07 | 2,308.0 |
| 21 | MLP | 2026-11-11 03:24:07 | 2026-11-11 03:40:32 | 985.5 |
| 17 | Frozen | 2026-11-11 03:40:32 | 2026-11-11 04:15:37 | 2,104.5 |
| 17 | RSSM | 2026-11-11 04:15:37 | 2026-11-11 05:21:37 | 3,959.9 |
| 18 | RSSM | 2026-11-11 05:21:37 | 2026-11-11 06:27:14 | 3,937.5 |
| 18 | Frozen | 2026-11-11 06:27:14 | 2026-11-11 07:05:02 | 2,268.0 |
| 19 | Frozen | 2026-11-11 07:05:02 | 2026-11-11 07:40:31 | 2,128.6 |
| 19 | RSSM | 2026-11-11 07:40:31 | 2026-11-11 08:47:25 | 4,014.6 |
| 20 | RSSM | 2026-11-11 08:47:25 | 2026-11-11 09:53:57 | 3,991.1 |
| 20 | Frozen | 2026-11-11 09:53:57 | 2026-11-11 10:29:24 | 2,127.5 |
| 21 | Frozen | 2026-11-11 10:29:24 | 2026-11-11 11:04:48 | 2,123.8 |
| 21 | RSSM | 2026-11-11 11:04:48 | 2026-11-11 12:11:19 | 3,990.7 |

Two seed-17 GRU attempts were superseded and excluded: 2026-11-10
20:00:06–20:13:20 UTC (793.9 s; stopped after the q-visible input design
review), and 20:14:31–20:14:34 UTC (2.8 s; zero-length commitment tensor
bug, fixed before any checkpoint). Their records and reasons are in the
training manifest.

The serial evaluation ledger is:

| Job | Started (UTC) | Finished (UTC) | Process seconds |
|:--|:--|:--|--:|
| GRU vs MLP | 2026-11-11 12:12:21 | 2026-11-11 12:29:15 | 1,013.5 |
| GRU/MLP grid visuals | 2026-11-11 12:29:15 | 2026-11-11 12:29:16 | 1.3 |
| 45-checkpoint convergence | 2026-11-11 12:29:16 | 2026-11-11 12:41:23 | 726.6 |
| Final quality and noisy-input VI | 2026-11-11 12:41:23 | 2026-11-11 13:24:25 | 2,582.1 |
| 50 policy maps | 2026-11-11 13:24:25 | 2026-11-11 13:24:47 | 22.4 |
| Final per-start return tables | 2026-11-11 13:34:34 | 2026-11-11 13:34:35 | 1.6 |

Training used CPU on macOS 26.6.2, arm64 Apple Silicon, Python 3.14.2,
PyTorch 2.14.0, and NumPy 2.5.2. Model metadata also preserves source hashes,
parameter counts, and exact training-loop timers.

Full-GRU throughput changed sharply during the serial campaign: seeds 17–19
took about 4,400 seconds, while seeds 20–21 took 2,717 and 2,308 seconds.
The full-GRU baseline cohort ran before the frozen/RSSM cohort, so machine
load or other time-dependent throughput can confound the apparent speed
difference. The paired savings are real measurements from these runs, but a
counterbalanced timing replication would be needed for a stronger causal
claim about freezing. This is the main remaining experiment if Section One
needs to claim a robust speed gain at preserved quality.

The 15 GRU/frozen/RSSM checkpoints were each evaluated for 100×19 rollouts at
up to 1,000 steps. The 15-policy final-quality pass took 2,582.1 seconds
(43m02s); the GRU/MLP pass took 1,013.5 seconds (16m54s); the 45-checkpoint
convergence pass took 726.6 seconds (12m07s); the 50-map policy gallery took
22.4 seconds; final return-table generation took 1.6 seconds. The evaluation
manifest records UTC start/end for every job.
All final data, full run timestamps, per-start rollouts, hyperparameters,
failed-attempt notes, and source hashes are in
[`training_manifest.json`](artifacts/q_observed_section_one/training_manifest.json),
[`evaluation_manifest.json`](artifacts/q_observed_section_one/evaluation_manifest.json),
and the result JSON files.

## Policy gallery and files

The existing notebook visualizer produced maps for each trained policy and
saved budget: [policy gallery](artifacts/q_observed_section_one/policy_maps/index.md)
(50 maps). The maps show the q₀ controller response under position noise;
colors represent expected max Q under the sensor channel, and arrows show
the modal greedy action. The task-grid empirical returns are shown separately
above.

Key reproducibility files:

- [Experiment plan and success criteria](RERUN_EXPERIMENT_PLAN.md)
- [Serial training manifest](artifacts/q_observed_section_one/training_manifest.json)
- [GRU-versus-MLP evaluation](artifacts/q_observed_section_one/history4_gru_vs_mlp.json)
- [Checkpoint convergence evaluation](artifacts/q_observed_section_one/convergence/evaluation.json)
- [Final quality and value-iteration evaluation](artifacts/q_observed_section_one/final_quality/evaluation.json)
- [Final per-start 5×4 return grids and tables](artifacts/q_observed_section_one/final_quality/grid_values.md)
- [Notebook-style final-grid generator](plot_q_observed_final_grids.py)
- [Job timing and command manifest](artifacts/q_observed_section_one/evaluation_manifest.json)
- [Older experiments and historical q-hidden results](RESEARCH_RESULTS_REPORT.md)

## Conclusion

With true `q` available and grid position noisy, the history-4 GRU-DQN
outperformed the matched current-observation MLP on all five training seeds,
with a positive paired interval. This answers the first question in favor of
recurrence. The experiment does not show the GRU outperforming noisy-input
value iteration. Freezing the GRU reduced measured wall time, but quality
preservation was inconclusive; the RSSM encoder did not produce a reliable
speed or quality advantage. A counterbalanced timing replication is needed
before making a strong training-speed claim.
