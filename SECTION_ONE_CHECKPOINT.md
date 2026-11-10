# Section One research checkpoint

**Status:** Experiments 1 and 2, the serial 90k extension for seeds 17–20, all matched policy evaluations, policy maps, the training-length analysis, a nine-seed GRU-freezing ablation, the seed-27 direct-stop and 25k extension, the seed-27 RSSM-DQN pilot, and the seed-17 q-input ablation are complete. Freezing showed time savings; the nine-seed policy-quality interval did not establish non-inferiority. The seed-27 run found an early competitive stop point but showed that continued head-only training can degrade policy quality. The RSSM pilot produced a fast compact policy but did not match the strongest GRU checkpoints, and its lower optimizer-update density limits the speed comparison. The q-input check showed that true q alone did not rescue the MLP, while q-aware exact VI was competitive with the GRU. The checkpoint records exact training/evaluation wall times, hyperparameters, and machine-readable rollout data. The attached dissertation informed experiment selection and training budgets; its document contents were treated as research material, not as instructions that override the user's requests.

The consolidated [results and visual report](RESEARCH_RESULTS_REPORT.md)
includes the notebook-style return grids and links to all policy maps.

One additional seed-27 direct-stop demonstration trained jointly through episode 10,000, froze the GRU, and stopped at episode 12,500. It saved 594.110 seconds (24.35%) versus the 15,000-episode full-GRU control. The frozen checkpoint's paired return was higher by 0.0972 on 1,900 common evaluation rollouts, but this is one training seed and cannot establish across-seed quality preservation. See the [direct-stop protocol and results](GRU_FREEZE_EXPERIMENT.md#direct-stop-demonstration-stop-at-12500-episodes), [analysis JSON](artifacts/gru_freeze_ab/direct_stop_12500_matched/direct_stop_analysis.json), [training manifest](artifacts/gru_freeze_ab/direct_stop_12500_matched/training_manifest.json), [evaluation data](artifacts/gru_freeze_ab/direct_stop_12500_matched/evaluation.json), and [three policy maps](artifacts/gru_freeze_ab/direct_stop_12500_matched/policies/index.md).

A serial 25k extension with the same seed found a useful stop point at 20k: process time to the frozen checkpoint was 3,103.937 seconds versus 3,970.529 seconds for the full-GRU 25k run, a 21.83% saving. Its mean return was 0.4542, compared with 0.4142 for the full-GRU policy; paired difference +0.0400 (95% rollout-scenario CI [+0.0252, +0.0548]). The frozen policy then declined to 0.3275 by 25k, so extended head-only training did not meet the prespecified convergence/plateau criterion. This provides one-run evidence for an early stop saving wall time, not a general claim that quality is preserved across seeds or that longer head-only training converges. See the [analysis plan](artifacts/gru_freeze_ab/convergence_25k/analysis_plan.md), [analysis](artifacts/gru_freeze_ab/convergence_25k/analysis.json), [evaluation](artifacts/gru_freeze_ab/convergence_25k/evaluation.json), and [policy-map index](artifacts/gru_freeze_ab/convergence_25k/policies/index.md).

The RSSM representation feeding the same DQN objective and Q head has now been piloted. See the [design and implementation notes](RSSM_DQN_DESIGN.md) and [full RSSM-DQN comparison report](RSSM_DQN_PILOT.md).

## Three-way encoder comparison: seed 27

The RSSM run trained for 5,000 episodes, with checkpoints at 1,000, 2,500,
3,750, and 5,000 episodes. All RSSM checkpoints were compared with the same
seed's full-GRU 25k and frozen-GRU 20k and 25k checkpoints, using 1,900 common
rollout scenarios per policy. The best RSSM checkpoint was at 3,750 episodes:
return 0.3589 in 454.4 seconds of training-loop time. It trailed full-GRU 25k
by 0.0553 (95% paired rollout CI [−0.0768, −0.0338]) and frozen-GRU 20k by
0.0953 ([−0.1170, −0.0735]); it exceeded frozen-GRU 25k by 0.0314
([+0.0082, +0.0546]). The RSSM also fell below the noisy-input MDP VI score
of 0.3989 by 0.0400 ([−0.0640, −0.0161]).

The strongest frozen-GRU checkpoint returned 0.4542 at 20k episodes, 0.0553
above noisy-input VI; full-GRU 25k returned 0.4142, with an interval spanning
zero for its +0.0153 VI difference. Frozen-GRU 25k returned 0.3275, below VI.
These results show substantial checkpoint sensitivity. In particular, RSSM
greedy return peaked at 3,750 and fell to 0.1952 by episode 5,000 even though
the average training return rose. The 3,750 RSSM checkpoint used much less
wall time (7.57 minutes), but it used one eighth the optimizer updates per
environment step of the GRU runs and had a shorter episode budget. It is not
an equal-update architecture comparison. The RSSM model had 2,979 parameters
versus 17,608 for the 64-unit GRU-DQN (83.1% fewer). All intervals are over
rollout scenarios from one training seed, so none estimates between-seed
variation.

The complete methods, exact UTC and process timing, model size and auxiliary
losses, paired results, caveats, and figures are in the [RSSM pilot report](RSSM_DQN_PILOT.md).
The raw [evaluation](artifacts/rssm_dqn_pilot/evaluation.json),
[analysis](artifacts/rssm_dqn_pilot/analysis.json), and [policy maps](artifacts/rssm_dqn_pilot/policies/index.md)
preserve the machine-readable data and grid visualizations.

## Research question and result

Experiment 1 asks whether a history-length-4 GRU-DQN can use partial observations to outperform (a) a matched feedforward DQN without history and (b) a conventional product-MDP value-iteration policy deployed by treating noisy readings as exact state indices. Across ten matched training seeds, GRU-DQN had higher return than the no-history MLP on all ten seeds and beat the noisy-input value-iteration policy on nine of ten at 15k episodes. At 90k, it beat noisy-input VI on all four evaluated seeds. The four-seed length check found mean returns of 0.482, 0.541, 0.585, and 0.552 at 15k, 30k, 45k, and 90k episodes. The mean 45k→90k change was −0.033 (95% Student-t CI −0.161 to +0.095), so doubling training did not produce a measurable average gain. The result establishes the comparison with noisy-input VI, but does not establish policy convergence.

Experiment 2 transfers the history-4 GRU-DQN to the one-agent office layout from Junchao Li's dissertation. The readout is derived from the full four-direction room-wall signature: unique signatures identify the current room; duplicate signatures create uniform aliases. There is no single-direction sensor. Across seven training seeds, the policy's mean probability of completing at least two safe LTL cycles was 0.114 (95% Student-t CI 0.004–0.224). This is positive evidence of transfer, with substantial seed variation and modest per-rollout success rates; it is not evidence of consistently high reliability.

The research summary and full value tables in the problem grids are in [EXPERIMENTS.md](EXPERIMENTS.md). The policy maps and routes are linked there and below.

## Additional training-cost ablation: freeze the GRU at 10,000 episodes

Nine paired seeds (18–26) compared full 15,000-episode GRU-DQN training with a schedule that trained the GRU and Q head jointly for 10,000 episodes, then froze the recurrent weights and trained only the Q head for 5,000 more. Both arms used 15,000 × 600 episodes/steps, history length 4, hidden size 64, batch size 32, replay capacity 100,000, AdamW at 1e-4, optimization every four environment steps, target updates every 50 steps, epsilon floor 0.1 reached by episode 3,750, and epsilon-action probability 0.2. Jobs ran serially with arm order alternated by seed.

The frozen schedule saved 209.820 seconds per paired run (95% Student-t CI 205.190–214.449), or 9.00% (95% CI 8.80–9.19%). Mean process wall time was 2,331.979 seconds for full training and 2,122.160 seconds for frozen training, about 0.1555 and 0.1415 seconds per episode, respectively. The final frozen-minus-full return contrast was −0.006 (95% CI −0.136 to +0.124), which fails the registered non-inferiority requirement that the lower bound exceed −0.05. Policy-quality preservation is therefore inconclusive even though the time-saving criterion passed. Mean final returns were 0.501 for frozen training and 0.507 for full training.

Training-prefix, online/target GRU-weight, and Q-head-change integrity checks passed for all nine seeds. The 36-checkpoint common-scenario evaluation took 5,871.590 seconds (1 h 37 min 52 s), using 100 rollouts per traversable start and a 1,000-step horizon. See the [full protocol and seed-level results](GRU_FREEZE_EXPERIMENT.md), [training manifest](artifacts/gru_freeze_ab/confirmatory_10k/training_manifest.json), [evaluation data](artifacts/gru_freeze_ab/confirmatory_10k/evaluation.json), [paired analysis](artifacts/gru_freeze_ab/confirmatory_10k/analysis.json), [comparison figure](artifacts/gru_freeze_ab/confirmatory_10k/comparison.png), and [36 policy maps](artifacts/gru_freeze_ab/confirmatory_10k/policies/index.md).

## Experiment 1: partially observable grid

### Design and training

- History-4 GRU-DQN and no-history, two-hidden-layer 64-unit feedforward MLP DQN; ten matched seeds 17–26.
- The GRU receives recent observations, previous actions, and its own remembered automaton epsilon target. The true automaton state is hidden. The MLP receives the current noisy position and Rabin-pair index; this task has only one pair, so the pair index is constant and does not reveal automaton progress.
- Training budget: 15,000 episodes × 600 steps, matching the closest thesis experiment (Li, 2024, Chapter 4, §4.5.1, Task 1) in episodes, steps per episode, batch size, and target-copy cadence; requested history length is four.
- Batch size 32; replay capacity 100,000; Adam learning rate 1e-4; replay optimization every four environment steps; target network copied every 50 environment steps; GRU hidden size 64; epsilon floor 0.1 after 25% of episodes; available automaton-transition action probability 0.2.
- Project environment retained its 5×4 map, transition probability 0.8 for intended motion and 0.25 position observation error. Discounts are 0.99999 outside accepting states and 0.99 in accepting states. The dissertation's Task 1 uses a 10×10 map, 0.9 intended motion, 0.1 observation error and an LSTM, so these are protocol-inspired, not identical environments.
- Training/evaluation ran on CPU. Multiple seed jobs ran concurrently for portions of the experiment; per-process wall times below are observed durations, not isolated benchmark throughput.

### Evaluation protocol and results

Each policy was evaluated greedily on paired scenarios: 100 rollouts from each of the 19 traversable starting cells, 1,000 steps per rollout. Reported confidence intervals for the overall comparison are Student-t intervals over the ten independent training seeds; paired per-seed intervals are rollout-level intervals.

| Comparison | Mean discounted return | 95% CI over training seeds |
|:---|---:|:---:|
| History-4 GRU-DQN | 0.51331 | [0.42711, 0.59951] |
| No-history MLP DQN | 0.08625 | [0.06390, 0.10860] |
| Paired GRU − MLP | +0.42706 | [+0.34780, +0.50632] |
| Paired GRU − MLP, starts not labeled `c` | +0.54094 | [+0.44055, +0.64134] |
| Noisy-observation MDP value iteration | 0.39890 | constant baseline |
| Paired GRU − noisy-observation MDP VI | +0.11441 | [+0.02821, +0.20062] |
| Paired GRU − noisy-observation MDP VI, non-`c` starts | +0.14492 | [+0.03573, +0.25411] |
| Full-state MDP VI oracle | 0.76462 | exact-state reference |

The conventional VI policy is computed on the exact product MDP. At evaluation it acts on the noisy location reading as though that reading were the true location; it does not maintain a belief state. It tracks its own automaton epsilon commitment. Full-state VI is an oracle reference, not the main baseline. The GRU beats noisy-input VI in nine seeds (seed 23 is the counterexample), and beats the matched MLP in all ten.

The seed-level results, per-cell grids, exact VI value grid, and plots are in [EXPERIMENTS.md](EXPERIMENTS.md), [the GRU/MLP evaluation JSON](artifacts/pomdp_history4_vs_feedforward_dqn_10seeds.json), and [the VI evaluation JSON](artifacts/pomdp_history4_vs_mdp_value_iteration_10seeds.json). The state-value and empirical return tables use the same 5×4 arrangement as the task map. The network is a Q-policy, so empirical per-start discounted returns are reported; they are not mislabeled as a separately learned scalar state-value function.

![Ten-seed paired gains over MDP value iteration indexed by noisy observations](artifacts/pomdp_policy_maps/gru_vs_noisy_mdp_vi_10seeds.png)

The wrong-index VI policy loses 0.366 mean return relative to full-state VI. The GRU adds 0.114 over wrong-index VI (descriptively closing about 31% of that mean loss), while remaining 0.251 below the full-state oracle. This supports the specific claim that a recurrent policy trained on partial observations improves on ordinary MDP value iteration when the latter is given noisy state indices. It does not establish superiority to a belief-state or other POMDP solver.

### Training-length diagnostic

Across the ten 15k runs, mean exploration-phase return rises from 0.155 in episodes 1–3k to 0.234 in episodes 3–6k, then stays near 0.229–0.237 through episode 15k. This noisy statistic is not a greedy-policy convergence check. Four matched 15k/30k/45k policies were evaluated on the same 1,900 scenarios per seed (100 rollouts from each of 19 traversable starts, 1,000-step horizon). Each length was trained from scratch with the same seed and epsilon-floor episode (3,750); all saved histories match exactly through shared episode prefixes.

| Training seed | 15k mean | 30k mean | 30k − 15k (95% rollout CI) | 45k mean | 45k − 30k (95% rollout CI) |
|---:|---:|---:|---:|---:|---:|
| 17 | 0.57160 | 0.59818 | +0.02658 (+0.00587, +0.04728) | 0.55093 | −0.04725 (−0.06703, −0.02747) |
| 18 | 0.43512 | 0.58435 | +0.14923 (+0.13098, +0.16748) | 0.60669 | +0.02234 (+0.00306, +0.04161) |
| 19 | 0.46097 | 0.52117 | +0.06020 (+0.04223, +0.07818) | 0.60859 | +0.08742 (+0.06851, +0.10633) |
| 20 | 0.46093 | 0.46138 | +0.00045 (−0.01238, +0.01328) | 0.57268 | +0.11130 (+0.09343, +0.12917) |
| **Mean across seeds** | **0.48215** | **0.54127** | **+0.05911 (−0.04408, +0.16231)** | **0.58472** | **+0.04345 (−0.06983, +0.15673)** |

Per-seed change intervals quantify paired rollout sampling error; bold change intervals are Student-t intervals over four independent training-seed differences. Three seeds improved from 30k to 45k, while seed 17 declined. The average 30k→45k gain is positive but uncertain; its cross-seed interval includes zero. The 45k mean is higher than 15k, but the paired 15k→45k change is heterogeneous and inconclusive (+0.10257, 95% CI −0.03387 to +0.23900). The later 90k comparison below adds another checkpoint, but the four-seed study still does not establish a reliable length effect or convergence.

The completed 90k evaluation gives a mean return of 0.55174 (95% Student-t CI
0.43318–0.67030), compared with 0.58472 at 45k. The paired mean 45k→90k
change is −0.03298 (95% CI −0.16061 to +0.09465). Per-seed changes were
−0.02567, +0.02056, −0.14865, and +0.02184. Two seeds improved slightly and
two declined, with seed 19 accounting for most of the average decline. The
training-return means across episodes 45,001–60,000, 60,001–75,000, and
75,001–90,000 were 0.21489, 0.21091, and 0.21431 across the four seeds, with
no consistent upward trend. Training return is exploration-phase behavior and
does not establish greedy-policy convergence. The rollout analysis likewise
does not prove convergence: it tests only 45k and 90k endpoints for four
training seeds, and the interval is wide enough to include a +0.095 return
gain (about 16% of the 45k mean) as well as a larger decline. More episodes
are not needed to support the comparison with noisy-input VI; these results
do not show that they would improve the average policy. To establish a
convergence claim, the next useful evidence would be more independent training
seeds and greedy evaluations at intermediate checkpoints.

At 90k, mean GRU return was 0.55174 versus 0.39890 for noisy-input VI; the
paired advantage was +0.15284 (95% CI +0.03428 to +0.27140 across the four
training seeds). Each seed's paired rollout interval was above zero. Full
per-seed results, evaluation timing, rollout samples, and the 45k/90k paired
analysis are documented in [EXPERIMENTS.md](EXPERIMENTS.md) and the linked
machine-readable reports. The 90k policies are shown in the policy-map table
there.

![Paired 15k, 30k, 45k, and 90k evaluation by training seed](artifacts/pomdp_policy_maps/training_length_15k_30k_45k_90k_seeds17_20.png)

## Experiment 2: single-agent office transfer

### Office map and analytical readout

The layout and proposition labels follow Figure 11 and the supplementary 4×4 office model in Li's thesis. The task is `G F (p & F (a | c)) & G !s`: repeatedly visit printer `p`, then `a` or `c`, while avoiding storage `s`. The agent starts at `(3,3)`. Motion follows the supplementary office model: probability 0.9 for an intended available direction; remaining probability shared across other available directions.

The abstracted `read_state` uses the complete N,S,E,W local signature with categories hallway=0, wall=1, door=2, window=3. For true room `s`, let `C(s)` be the set of rooms with the same signature. Then `P(o|s)=1/|C(s)|` for `o∈C(s)` and 0 otherwise. A unique signature returns the room exactly. Duplicate signatures are uniform aliases. The separate thesis helper that samples a single direction is not used; this readout rule is the project's analytical observation abstraction, not a claim that Li used the same probability distribution.

- Signature `(wall, wall, door, window)` aliases `(0,0)` and `(3,0)`, giving probability 1/2 per candidate.
- Signature `(wall, wall, door, wall)` aliases `(1,0)`, `(2,0)`, and `(2,2)`, giving probability 1/3 per candidate.
- The other eleven locations have unique signatures and return probability 1.
- The exact 16×16 observation-probability matrix and signature groups are stored in each office checkpoint's JSON metadata. `office_pomdp.py` implements the alias calculation; `train_office_pomdp.py` stores the resulting matrix in run metadata.

### Training and evaluation

The main group is seven independent one-agent runs (seeds 22–28), each 10,000 episodes × 300 steps. Hyperparameters: history length 4, GRU hidden size 64, batch size 32, replay capacity 100,000, learning rate 1e-4, replay optimizer every four environment steps, target copy every 50 environment steps, epsilon floor 0.1 reached after 25% of training, automaton-transition action probability 0.2, discounts 0.98 for nonaccepting states and 0.99 for accepting states. The 0.98 nonaccepting discount matches the thesis, while 0.99 preserves this project's accepting-state convention. Each run performs 3,000,000 environment steps, approximately 750,000 replay optimizer updates, and 60,000 target-network copies (assuming all environment steps are counted from zero as in the trainer).

Each trained policy was evaluated at the thesis start and every office cell with 100 rollouts per start, 1,000-step horizon. The task-success metric counts actual LDBA acceptance reward events and separately requires no storage violation for “safe cycle” metrics. Physical `Print → a/c` label sequences are only a diagnostic and are not treated as accepted LTL cycles. Per-seed rollout success probabilities use Wilson 95% intervals; across-training-seed intervals use Student-t intervals.

| Seed | Mean accepted cycles | P(≥1 safe cycle) | P(≥2 safe cycles) | No storage violation | Mean discounted return |
|---:|---:|---:|---:|---:|---:|
| 22 | 1.73 | 0.51 | 0.35 | 0.56 | 0.001636 |
| 23 | 0.93 | 0.22 | 0.06 | 0.28 | 0.001772 |
| 24 | 0.39 | 0.20 | 0.08 | 0.60 | 0.000643 |
| 25 | 0.80 | 0.57 | 0.19 | 0.94 | 0.000090 |
| 26 | 0.45 | 0.31 | 0.07 | 0.83 | 0.001325 |
| 27 | 0.15 | 0.05 | 0.00 | 0.47 | 0.000228 |
| 28 | 0.24 | 0.17 | 0.05 | 0.73 | 0.000167 |
| **Across-seed mean** | **0.67** | **0.29** | **0.114** | **0.63** | **0.000837** |
| **95% Student-t CI** | **[0.165, 1.175]** | **[0.116, 0.464]** | **[0.004, 0.224]** | **[0.423, 0.837]** | **[0.000165, 0.001509]** |

The seven-seed mean and intervals support transfer and repeated safe completion at the group level. Seed variation is large: P(≥2 safe cycles) ranges 0–0.35 and no-storage-violation rates range 0.28–0.94. The group does not establish dependable high success for an individual training seed. Five earlier diagnostic runs (seeds 17–21, nonaccepting discount 0.99999) are reported separately in `EXPERIMENTS.md`; their mean P(≥2 safe cycles) was 0.09 with 95% CI −0.16 to 0.34, and they are not pooled into the main transfer group.

### Office figures and grid values

- [Mean return, mean accepted cycles, and safe two-cycle probability arranged on the 4×4 map](artifacts/office_policy_maps/office_gamma098_grid_7seeds.png)
- [Seven-seed office training returns](artifacts/office_policy_maps/training_returns_gamma098_7seeds.png)
- [Full office policy/action maps and thesis-start routes, seeds 17–28](EXPERIMENTS.md#experiment-2-transfer-to-lis-office-environment)
- Exact full-state product-MDP value-iteration grid and architectural-signature grid appear in `EXPERIMENTS.md`. Neural maps show max action value and modal greedy action per physical cell, along with representative rollouts; the heatmap is a visualization of Q values, not a calibrated state-value estimate.

## Training timing provenance

The timestamp columns are UTC as written by the trainers. Seconds per episode are total process wall seconds divided by the configured episode count; the code does not record a separate timer for each individual episode. Durations include machine contention where seed runs overlapped. The evaluation JSON does not include per-seed elapsed timing for Experiment 1 comparisons, so no such durations are inferred. Office main-group evaluation timings are recorded below.

### Experiment 1 GRU-DQN, 15,000 episodes

| Seed | Episodes × steps | Started (UTC) | Finished (UTC) | Wall seconds | Seconds/episode | Mean return | First 1,000 | Last 1,000 |
|---:|:---:|:---|:---|---:|---:|---:|---:|---:|
| 17 | 15000 × 600 | 2026-11-03T19:43:46.693870+00:00 | 2026-11-03T20:56:35.458514+00:00 | 4368.754 | 0.29125 | 0.225912 | 0.120838 | 0.233509 |
| 18 | 15000 × 600 | 2026-11-03T20:56:47.733928+00:00 | 2026-11-03T22:13:17.389241+00:00 | 4589.694 | 0.30598 | 0.221115 | 0.123850 | 0.253427 |
| 19 | 15000 × 600 | 2026-11-03T22:32:04.034577+00:00 | 2026-11-03T23:44:55.105332+00:00 | 4371.063 | 0.29140 | 0.203087 | 0.120818 | 0.209511 |
| 20 | 15000 × 600 | 2026-11-04T02:39:19.683592+00:00 | 2026-11-04T04:09:44.392106+00:00 | 5424.715 | 0.36165 | 0.213277 | 0.106902 | 0.208583 |
| 21 | 15000 × 600 | 2026-11-04T02:39:19.683601+00:00 | 2026-11-04T04:09:36.977967+00:00 | 5417.301 | 0.36115 | 0.208404 | 0.126842 | 0.204541 |
| 22 | 15000 × 600 | 2026-11-04T02:39:19.683619+00:00 | 2026-11-04T04:09:55.636081+00:00 | 5435.958 | 0.36240 | 0.213736 | 0.129844 | 0.208548 |
| 23 | 15000 × 600 | 2026-11-04T02:45:03.026066+00:00 | 2026-11-04T04:17:57.695270+00:00 | 5574.672 | 0.37164 | 0.219603 | 0.133845 | 0.248461 |
| 24 | 15000 × 600 | 2026-11-04T02:45:03.025904+00:00 | 2026-11-04T04:18:32.447986+00:00 | 5609.425 | 0.37396 | 0.240140 | 0.129835 | 0.270395 |
| 25 | 15000 × 600 | 2026-11-04T04:28:17.130709+00:00 | 2026-11-04T06:03:29.265456+00:00 | 5712.080 | 0.38081 | 0.215201 | 0.124778 | 0.268375 |
| 26 | 15000 × 600 | 2026-11-04T04:28:17.130704+00:00 | 2026-11-04T06:04:02.990424+00:00 | 5745.805 | 0.38305 | 0.218795 | 0.124817 | 0.247484 |

### Experiment 1 no-history MLP, 15,000 episodes

| Seed | Episodes × steps | Started (UTC) | Finished (UTC) | Wall seconds | Seconds/episode | Mean return | First 1,000 | Last 1,000 |
|---:|:---:|:---|:---|---:|---:|---:|---:|---:|
| 17 | 15000 × 600 | 2026-11-04T05:06:02.607124+00:00 | 2026-11-04T06:15:36.919361+00:00 | 4174.272 | 0.27828 | 0.088526 | 0.104000 | 0.086973 |
| 18 | 15000 × 600 | 2026-11-04T05:06:02.783528+00:00 | 2026-11-04T06:15:33.030228+00:00 | 4170.206 | 0.27801 | 0.091587 | 0.115146 | 0.089937 |
| 19 | 15000 × 600 | 2026-11-04T05:06:02.782586+00:00 | 2026-11-04T06:15:36.105386+00:00 | 4173.283 | 0.27822 | 0.089182 | 0.104977 | 0.090478 |
| 20 | 15000 × 600 | 2026-11-04T05:06:02.745591+00:00 | 2026-11-04T06:15:23.006336+00:00 | 4160.221 | 0.27735 | 0.088960 | 0.105062 | 0.075031 |
| 21 | 15000 × 600 | 2026-11-04T05:06:02.773397+00:00 | 2026-11-04T06:15:22.772126+00:00 | 4159.959 | 0.27733 | 0.089916 | 0.097163 | 0.094456 |
| 22 | 15000 × 600 | 2026-11-04T05:06:02.736147+00:00 | 2026-11-04T06:15:34.405852+00:00 | 4171.629 | 0.27811 | 0.085964 | 0.116169 | 0.093094 |
| 23 | 15000 × 600 | 2026-11-04T05:06:02.759651+00:00 | 2026-11-04T06:15:17.014523+00:00 | 4154.215 | 0.27695 | 0.091603 | 0.123019 | 0.091705 |
| 24 | 15000 × 600 | 2026-11-04T05:06:02.817669+00:00 | 2026-11-04T06:15:17.781308+00:00 | 4154.923 | 0.27699 | 0.092787 | 0.128937 | 0.102131 |
| 25 | 15000 × 600 | 2026-11-04T06:04:12.977897+00:00 | 2026-11-04T06:52:41.903521+00:00 | 2908.943 | 0.19393 | 0.089796 | 0.110932 | 0.094334 |
| 26 | 15000 × 600 | 2026-11-04T06:04:13.061057+00:00 | 2026-11-04T06:52:26.376904+00:00 | 2893.334 | 0.19289 | 0.085969 | 0.105112 | 0.090265 |

### Experiment 1 GRU-DQN, 30,000-episode extended-length runs

| Seed | Episodes × steps | Started (UTC) | Finished (UTC) | Wall seconds | Seconds/episode | Mean return | First 1,000 | Last 1,000 |
|---:|:---:|:---|:---|---:|---:|---:|---:|---:|
| 17 | 30000 × 600 | 2026-11-04T02:39:19.683596+00:00 | 2026-11-04T05:18:04.054738+00:00 | 9524.348 | 0.31748 | 0.230293 | 0.120838 | 0.202534 |
| 18 | 30000 × 600 | 2026-11-04T08:44:14.807188+00:00 | 2026-11-04T10:34:34.383250+00:00 | 6619.517 | 0.22065 | 0.231318 | 0.123850 | 0.224512 |
| 19 | 30000 × 600 | 2026-11-04T08:44:14.807191+00:00 | 2026-11-04T10:35:24.101197+00:00 | 6669.234 | 0.22231 | 0.212425 | 0.120818 | 0.210517 |
| 20 | 30000 × 600 | 2026-11-04T08:44:14.807188+00:00 | 2026-11-04T10:35:18.291625+00:00 | 6663.425 | 0.22211 | 0.223642 | 0.106902 | 0.217540 |

### Experiment 1 GRU-DQN, 45,000-episode matched runs

All four runs were trained from scratch with seeds 17–20, 45,000 episodes × 600 steps, and epsilon-decay fraction 0.0833333 so epsilon reached its 0.1 floor at episode 3,750. Other settings match the 15k protocol: history 4, GRU hidden size 64, batch 32, replay capacity 100,000, learning rate 1e-4, replay optimization every four environment steps, target copy every 50 steps, and automaton-action probability 0.2. Each run used 27,000,000 environment steps, made 6,750,000 replay-optimizer calls (the initial calls skip until the 32-item batch is available), and copied target weights exactly 540,000 times. The saved episode-return histories match the corresponding 15k and 30k runs exactly over their shared prefixes. The four jobs started within 2 milliseconds of one another, so the elapsed times include shared CPU contention.

| Seed | Episodes × steps | Started (UTC) | Finished (UTC) | Wall seconds | Seconds/episode | Mean return | First 1,000 | Last 1,000 |
|---:|:---:|:---|:---|---:|---:|---:|---:|---:|
| 17 | 45000 × 600 | 2026-11-04T11:24:00.035432+00:00 | 2026-11-04T14:43:40.556937+00:00 | 11980.494 | 0.26623 | 0.226100 | 0.120838 | 0.219516 |
| 18 | 45000 × 600 | 2026-11-04T11:24:00.037096+00:00 | 2026-11-04T14:43:54.542257+00:00 | 11994.478 | 0.26654 | 0.227030 | 0.123850 | 0.227493 |
| 19 | 45000 × 600 | 2026-11-04T11:24:00.035475+00:00 | 2026-11-04T14:43:15.967577+00:00 | 11955.905 | 0.26569 | 0.208888 | 0.120818 | 0.222486 |
| 20 | 45000 × 600 | 2026-11-04T11:24:00.035419+00:00 | 2026-11-04T14:41:04.096890+00:00 | 11824.035 | 0.26276 | 0.227825 | 0.106902 | 0.237462 |

Each 45k checkpoint has a policy visualization in `artifacts/pomdp_policy_maps/seed<N>_gru_45k.png`; the maps use the same notebook-style task-grid layout as the 15k and 30k figures.

### 90,000-episode extension and convergence check

The 45k result was heterogeneous (three seeds improved from 30k to 45k, one declined) and the four-seed interval remained wide. To check whether 45k episodes were too few, seeds 17–20 were trained to 90,000 episodes × 600 steps (54,000,000 environment steps per seed). These are the same four matched seeds used at 15k/30k/45k, so the extension compares training length within each seed; four remains a small sample and does not make the across-seed uncertainty precise. These jobs ran one at a time following the user's CPU-throughput preference. They use history 4, GRU hidden size 64, batch 32, replay capacity 100,000, learning rate 1e-4, replay optimizer every four environment steps, target copy every 50 steps, and automaton-action probability 0.2. This schedules 13,500,000 replay-optimizer calls (the initial calls skip until the 32-item batch is available) and exactly 1,080,000 target copies per run. Epsilon reaches 0.1 at episode 3,750 (`epsilon_decay_fraction=0.0416666667`). The 90k runs started from scratch; matching seeds and the fixed epsilon-floor episode make their first 45k episode-return histories directly comparable with the completed checkpoints.

| Seed | Started (UTC) | Finished (UTC) | Wall seconds | Seconds/episode | Mean training return | First 1,000 | Last 1,000 | 45k prefix matches |
|---:|:---|:---|---:|---:|---:|---:|---:|:---:|
| 17 | 2026-11-04T15:41:03.597639+00:00 | 2026-11-04T21:49:59.105539+00:00 | 22135.490 | 0.2460 | 0.219556 | 0.120838 | 0.214507 | Yes |
| 18 | 2026-11-04T21:50:01.177669+00:00 | 2026-11-05T01:49:40.347186+00:00 | 14379.168 | 0.1598 | 0.219078 | 0.123850 | 0.230459 | Yes |
| 19 | 2026-11-05T01:49:42.353573+00:00 | 2026-11-05T05:49:51.432623+00:00 | 14409.009 | 0.1601 | 0.208128 | 0.120818 | 0.230448 | Yes |
| 20 | 2026-11-05T05:49:53.407544+00:00 | 2026-11-05T09:49:15.862328+00:00 | 14362.427 | 0.1596 | 0.224898 | 0.106902 | 0.214486 | Yes |

All four completed episode histories match the corresponding 45k histories exactly over the first 45,000 episodes. Seed 20's run took 14,362.427 seconds (3 h 59 min 22 s; 0.1596 seconds per episode). All four policy maps and the common-scenario evaluation are complete. Training ran serially, as requested. Exact per-run training and evaluation timings appear in [EXPERIMENTS.md](EXPERIMENTS.md#90k-extension-and-convergence-check) and the timed JSON reports.

Exploration-return means in episodes 45,001–60,000, 60,001–75,000, and 75,001–90,000 were 0.21412/0.21385/0.21106 for seed 17, 0.22582/0.19649/0.21106 for seed 18, 0.19943/0.20634/0.21633 for seed 19, and 0.22017/0.22696/0.21878 for seed 20 (four-seed means 0.21489/0.21091/0.21431). They show no consistent upward trend in training returns after 45k. They are not a policy-convergence test: epsilon remains 0.1, the automaton-transition action probability remains 0.2, and rewards are sparse. The matched greedy-policy evaluation is complete: mean return changes by −0.03298 from 45k to 90k, with 95% CI [−0.16061, +0.09465]. It does not show that more training improves policy quality, and its uncertainty is too wide to establish convergence.

### Paired 15k/30k evaluation timing

All eight checkpoints used evaluation seed 17, 100 rollouts from each of 19
starts, and a 1,000-step horizon. The batch ran from 2026-11-04 10:59:53.905
UTC to 11:21:26.014 UTC (1,292.110 seconds total).

| Training seed | Episodes | Eval started (UTC) | Eval finished (UTC) | Eval wall seconds |
|---:|---:|:---|:---|---:|
| 17 | 15,000 | 2026-11-04T10:59:53.905597+00:00 | 2026-11-04T11:02:35.507509+00:00 | 161.600 |
| 18 | 15,000 | 2026-11-04T11:02:35.507963+00:00 | 2026-11-04T11:05:17.031888+00:00 | 161.534 |
| 19 | 15,000 | 2026-11-04T11:05:17.032330+00:00 | 2026-11-04T11:07:58.738679+00:00 | 161.705 |
| 20 | 15,000 | 2026-11-04T11:07:58.739141+00:00 | 2026-11-04T11:10:39.817163+00:00 | 161.077 |
| 17 | 30,000 | 2026-11-04T11:10:39.817619+00:00 | 2026-11-04T11:13:21.904208+00:00 | 162.086 |
| 18 | 30,000 | 2026-11-04T11:13:21.904674+00:00 | 2026-11-04T11:16:03.583088+00:00 | 161.677 |
| 19 | 30,000 | 2026-11-04T11:16:03.583550+00:00 | 2026-11-04T11:18:44.818101+00:00 | 161.234 |
| 20 | 30,000 | 2026-11-04T11:18:44.818554+00:00 | 2026-11-04T11:21:26.013480+00:00 | 161.194 |

The timed [15k/30k evaluation report](artifacts/pomdp_history4_length_eval_seeds17_20_timed.json) stores per-rollout returns and exact timing metadata; the [analysis report](artifacts/pomdp_history4_length_comparison_seeds17_20.json) and [paired figure](artifacts/pomdp_policy_maps/training_length_15k_vs_30k_seeds17_20.png) contain the summarized comparison. Each trained 30k policy has its own map linked in `EXPERIMENTS.md`.

### Paired 15k/30k/45k evaluation timing

All 12 checkpoints used evaluation seed 17, 100 rollouts from each of 19 traversable starts (1,900 rollouts per checkpoint), and a 1,000-step horizon. The batch took 3,162.126 seconds (52 min 42 s), from 2026-11-04T14:45:27.532529Z to 2026-11-04T15:38:09.658330Z.

| Training seed | Episodes | Eval started (UTC) | Eval finished (UTC) | Eval wall seconds |
|---:|---:|:---|:---|---:|
| 17 | 15,000 | 2026-11-04T14:45:27.532529+00:00 | 2026-11-04T14:48:11.225095+00:00 | 163.692 |
| 18 | 15,000 | 2026-11-04T14:48:11.225549+00:00 | 2026-11-04T14:50:54.057954+00:00 | 162.832 |
| 19 | 15,000 | 2026-11-04T14:50:54.058424+00:00 | 2026-11-04T14:53:39.857844+00:00 | 165.799 |
| 20 | 15,000 | 2026-11-04T14:53:39.858354+00:00 | 2026-11-04T14:56:46.517105+00:00 | 186.658 |
| 17 | 30,000 | 2026-11-04T14:56:46.518159+00:00 | 2026-11-04T15:01:56.561954+00:00 | 310.043 |
| 18 | 30,000 | 2026-11-04T15:01:56.562991+00:00 | 2026-11-04T15:07:05.883563+00:00 | 309.320 |
| 19 | 30,000 | 2026-11-04T15:07:05.884640+00:00 | 2026-11-04T15:12:14.376303+00:00 | 308.491 |
| 20 | 30,000 | 2026-11-04T15:12:14.377325+00:00 | 2026-11-04T15:17:22.968535+00:00 | 308.590 |
| 17 | 45,000 | 2026-11-04T15:17:22.969610+00:00 | 2026-11-04T15:22:31.559640+00:00 | 308.589 |
| 18 | 45,000 | 2026-11-04T15:22:31.560708+00:00 | 2026-11-04T15:27:40.577384+00:00 | 309.016 |
| 19 | 45,000 | 2026-11-04T15:27:40.578427+00:00 | 2026-11-04T15:32:53.993430+00:00 | 313.441 |
| 20 | 45,000 | 2026-11-04T15:32:53.994384+00:00 | 2026-11-04T15:38:09.658330+00:00 | 315.665 |

Exact per-rollout results and timing metadata are retained in [the 12-checkpoint JSON](artifacts/pomdp_history4_length_eval_seeds17_20_15k_30k_45k_timed.json), with [paired length analysis](artifacts/pomdp_history4_length_comparison_seeds17_20_15k_30k_45k.json) and [figure](artifacts/pomdp_policy_maps/training_length_15k_30k_45k_seeds17_20.png).

### Office diagnostic group, seeds 17–21

| Seed | Episodes × steps | Started (UTC) | Finished (UTC) | Wall seconds | Seconds/episode | Mean return | First 1,000 | Last 1,000 |
|---:|:---:|:---|:---|---:|---:|---:|---:|---:|
| 17 | 10000 × 300 | 2026-11-04T06:29:18.789761+00:00 | 2026-11-04T07:04:46.167051+00:00 | 2127.383 | 0.21274 | 0.002581 | 0.004868 | 0.001178 |
| 18 | 10000 × 300 | 2026-11-04T06:29:18.791220+00:00 | 2026-11-04T07:04:46.364111+00:00 | 2127.579 | 0.21276 | 0.002549 | 0.004290 | 0.002175 |
| 19 | 10000 × 300 | 2026-11-04T06:29:18.792804+00:00 | 2026-11-04T07:04:50.334629+00:00 | 2131.548 | 0.21315 | 0.003090 | 0.004769 | 0.002455 |
| 20 | 10000 × 300 | 2026-11-04T07:13:25.485542+00:00 | 2026-11-04T07:31:24.184621+00:00 | 1078.674 | 0.10787 | 0.003802 | 0.004897 | 0.003234 |
| 21 | 10000 × 300 | 2026-11-04T07:13:25.485615+00:00 | 2026-11-04T07:31:24.101924+00:00 | 1078.592 | 0.10786 | 0.003014 | 0.004976 | 0.002275 |
### Office main transfer group, seeds 22–28

| Seed | Episodes × steps | Started (UTC) | Finished (UTC) | Wall seconds | Seconds/episode | Mean return | First 1,000 | Last 1,000 |
|---:|:---:|:---|:---|---:|---:|---:|---:|---:|
| 22 | 10000 × 300 | 2026-11-04T07:34:09.755288+00:00 | 2026-11-04T07:54:17.401496+00:00 | 1207.654 | 0.12077 | 0.000968 | 0.000820 | 0.001208 |
| 23 | 10000 × 300 | 2026-11-04T07:34:09.750161+00:00 | 2026-11-04T07:54:15.867268+00:00 | 1206.125 | 0.12061 | 0.001000 | 0.000894 | 0.001395 |
| 24 | 10000 × 300 | 2026-11-04T07:34:09.753462+00:00 | 2026-11-04T07:54:18.087353+00:00 | 1208.342 | 0.12083 | 0.000717 | 0.000927 | 0.000870 |
| 25 | 10000 × 300 | 2026-11-04T07:58:34.202462+00:00 | 2026-11-04T08:13:50.633264+00:00 | 916.454 | 0.09165 | 0.001242 | 0.000845 | 0.001604 |
| 26 | 10000 × 300 | 2026-11-04T07:58:34.202462+00:00 | 2026-11-04T08:13:49.922560+00:00 | 915.743 | 0.09157 | 0.000687 | 0.000806 | 0.000809 |
| 27 | 10000 × 300 | 2026-11-04T08:18:54.573795+00:00 | 2026-11-04T08:33:47.495630+00:00 | 892.904 | 0.08929 | 0.000708 | 0.000884 | 0.000939 |
| 28 | 10000 × 300 | 2026-11-04T08:18:54.573883+00:00 | 2026-11-04T08:33:46.327836+00:00 | 891.737 | 0.08917 | 0.000802 | 0.000805 | 0.000974 |

### Office main-group evaluation wall time

| Training seed | Eval started (UTC) | Eval finished (UTC) | Eval wall seconds |
|---:|:---|:---|---:|
| 22 | 2026-11-04T07:54:31.219250+00:00 | 2026-11-04T07:56:04.261164+00:00 | 93.042 |
| 23 | 2026-11-04T07:56:04.262163+00:00 | 2026-11-04T07:57:20.594891+00:00 | 76.333 |
| 24 | 2026-11-04T07:57:20.595663+00:00 | 2026-11-04T07:58:56.729073+00:00 | 96.133 |
| 25 | 2026-11-04T08:14:27.836300+00:00 | 2026-11-04T08:16:22.353055+00:00 | 114.517 |
| 26 | 2026-11-04T08:16:22.354197+00:00 | 2026-11-04T08:18:09.721579+00:00 | 107.368 |
| 27 | 2026-11-04T08:34:04.635544+00:00 | 2026-11-04T08:35:25.488208+00:00 | 80.850 |
| 28 | 2026-11-04T08:35:25.489114+00:00 | 2026-11-04T08:37:03.566318+00:00 | 98.077 |

## Automaton-state input ablation: seed 17

The original MLP saw only current noisy position and Rabin-pair index. This
task has one Rabin pair, so that index is constant and conveys no automaton
progress. The GRU also
hides the true automaton state; it sees recent observations, previous actions,
and its remembered epsilon target. A matched 15,000-episode MLP run added true
q while retaining the noisy position and the same global action interface.
Training took 1,820.729 s (30 min 20.729 s); evaluation of 1,900 common
rollouts took 538.833 s (8 min 58.833 s).

The history-4 GRU returned 0.57160, the no-q MLP 0.10526, and the q-observed
MLP 0.10368. Adding q changed the MLP return by −0.00158 (paired 95% rollout
CI [−0.02097, +0.01780]). Exact value iteration given true q but noisy
position returned 0.53306; the GRU's paired gain was +0.03853
([+0.01546, +0.06160]). The q-aware MLP's training return stayed near 0.11,
indicating that this DQN run failed to learn the strong q-conditioned policy
shown by exact VI. These are one-seed results; their rollout intervals do not
measure training-seed variation. The earlier MLP run with a true-state legal
action mask scored 0.21210 but changed more than q and is treated as
exploratory.

See the full [methods, timing, per-start 5×4 return tables, and analysis](POMDP_AUTOMATON_STATE_ABLATION.md),
[matched rollout data](artifacts/pomdp_automaton_state_ablation/evaluation_q_observed.json),
and [task-grid return/action maps](artifacts/pomdp_automaton_state_ablation/index.md).

## Sources and reproducibility

- Li, Junchao. *Reinforcement Learning-Based Motion Planning in Partially Observable Environments for Complex Tasks* (2024 dissertation), especially Chapter 4, §4.5.1, Task 1, and Figure 11. Source PDF supplied by the user: `Reinforcement Learning-Based Motion Planning in Partially Observable Environments for Complex Tasks.pdf` in Downloads.
- [Dissertation record and DOI](https://doi.org/10.25820/etd.007512).
- [Supplementary office POMDP model](https://github.com/JunchaoLi001/LDGBA-Model_Checking/blob/main/Office%20Simulation/Office%20single%20obsv%20by%20SARSOP/csrl/pomdp.py) and [product-POMDP notebook](https://github.com/JunchaoLi001/LDGBA-Model_Checking/blob/main/Office%20Simulation/Office%20single%20obsv%20by%20SARSOP/generate%20product%20POMDP.ipynb).
- Code and outputs are in the current workspace. Git base revision: `918ef9ee09946cbbb424a3e736d665bf7423a524`; worktree was dirty during experiments.
- Training machine: macOS 26.6.2, arm64 Apple Silicon, Python 3.14.2, PyTorch 2.14.0, NumPy 2.5.2; CPU training.
- Reproduction commands, all experiment configuration details, seed-level comparison tables, full 5×4 and 4×4 value/return grids, observation signatures, and figure links are in [EXPERIMENTS.md](EXPERIMENTS.md).
- No software tests were run for this research checkpoint; the user did not request a test run. Training and evaluation jobs described above were executed, and visualization generation completed.
