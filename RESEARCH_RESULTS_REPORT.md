# Section One: experiments and results

## Executive summary

This report brings together the completed partially observable grid, office
transfer, GRU-freezing, and RSSM-DQN experiments. The headline results are:

| Study | Main result | Scope |
|:--|:--|:--|
| History-4 GRU versus no-history MLP | GRU mean return 0.513 versus 0.086; paired gain +0.427 (95% CI +0.348 to +0.506). GRU won on all 10 matched training seeds. | 15k episodes per seed; 10 seeds |
| History-4 GRU versus noisy-input MDP value iteration | Paired gain +0.114 (95% CI +0.028 to +0.201); GRU won on 9/10 seeds. | Same task and 10 seeds |
| Training-length check | Mean return rose through 45k, then was 0.033 lower at 90k; the four-seed interval included no change. | 15k–90k; 4 seeds |
| One-agent office transfer | Mean probability of completing at least two safe LTL cycles: 0.114 (95% CI 0.004–0.224). | 10k episodes; 7 seeds |
| GRU freezing | Saved about 9% wall time across nine paired seeds; quality preservation was inconclusive under the registered margin. | 15k episodes; 9 seeds |
| RSSM-inspired encoder into DQN | Best checkpoint returned 0.359 after 7.57 minutes; below the strongest GRU checkpoints and noisy-input VI. | 5k-episode pilot; 1 seed |

The strongest supported Section One claim is specific: a history-4 GRU-DQN
trained on partial observations substantially improves over a matched
no-history DQN and improves over regular MDP value iteration when that policy
is indexed using noisy observations. The office experiment provides
seven-seed transfer evidence with modest success rates and wide seed
variation. GRU freezing saves time, but the confirmatory quality result is
inconclusive. The RSSM run is a compact-model feasibility pilot, not evidence
that an RSSM encoder matches GRU quality.

## Visual guide

The [notebook-style grid gallery](artifacts/research_summary/grid_visualizations/index.md)
contains 23 grid figures rendered through the same `GridMDP.plot` code used in
the notebooks. Numbers show empirical discounted returns from each start
square or paired differences by square. They are not learned scalar state
values. The original grid labels, walls, traps, axes, and visual conventions
are retained.

### Core partially observable task

The main comparison trains a history-4 GRU-DQN and a matched no-history MLP
for 15,000 episodes × 600 steps on the 5×4 task. Evaluation uses 100 rollouts
from each of 19 traversable starts and a 1,000-step horizon. The automaton
state is hidden from both learned policies.

![Paired ten-seed GRU and no-history MLP comparison](artifacts/pomdp_policy_maps/gru_vs_mlp_10seeds.png)

![GRU-DQN versus MDP value iteration indexed by noisy observations](artifacts/pomdp_policy_maps/gru_vs_noisy_mdp_vi_10seeds.png)

![Ten-seed training return curves](artifacts/pomdp_policy_maps/training_returns_10seeds.png)

The per-start return grids provide the problem-layout view: [GRU](artifacts/research_summary/grid_visualizations/gru_15k_10seeds.png),
[no-history MLP](artifacts/research_summary/grid_visualizations/mlp_15k_10seeds.png),
[noisy-input MDP value iteration](artifacts/research_summary/grid_visualizations/noisy_vi_10seeds.png),
[full-state MDP reference](artifacts/research_summary/grid_visualizations/full_state_vi_10seeds.png),
[GRU minus MLP](artifacts/research_summary/grid_visualizations/gru_minus_mlp_by_start.png),
and [GRU minus noisy-input VI](artifacts/research_summary/grid_visualizations/gru_minus_noisy_vi_by_start.png).

The ten-seed mean returns were 0.51331 for GRU-DQN, 0.08625 for the MLP, and
0.39890 for noisy-input MDP value iteration. The full-state MDP policy scored
0.76462 and is an oracle reference. The paired GRU-minus-MLP gain was +0.42706
(95% Student-t CI [+0.34780, +0.50632]); all ten seeds favored the GRU. The
paired gain over noisy-input VI was +0.11441 (95% CI [+0.02821, +0.20062]);
nine of ten seeds favored the GRU. This does not compare against a belief-state
POMDP solver.

### Training length

Four matched seeds were evaluated at 15k, 30k, 45k, and 90k episodes. The
training-length plot is below; per-start grids for the longer checkpoints are
available in the [grid gallery](artifacts/research_summary/grid_visualizations/index.md).

![Paired return by training length, 15k to 90k episodes](artifacts/pomdp_policy_maps/training_length_15k_30k_45k_90k_seeds17_20.png)

Mean returns were 0.482, 0.541, 0.585, and 0.552 at those budgets. The paired
45k-to-90k change was −0.033 (95% CI [−0.161, +0.095]). The endpoint data do
not establish a length benefit or convergence.

### Transfer to the one-agent office environment

The office experiment uses the thesis office layout and an analytical
observation rule based on complete local wall-signature uniqueness. It trains
history-4 GRU-DQN for 10,000 episodes × 300 steps across seven seeds.

![Office mean return, accepted cycles, and safe two-cycle probability on the 4×4 map](artifacts/office_policy_maps/office_gamma098_grid_7seeds.png)

![Office training-return curves](artifacts/office_policy_maps/training_returns_gamma098_7seeds.png)

The across-seed probability of completing at least two safe LTL cycles was
0.114 (95% Student-t CI [0.004, 0.224]). The result supports transfer, with
substantial variability and modest success rates; it does not show dependable
high success from every training seed. The [office policy maps and routes](EXPERIMENTS.md#experiment-2-transfer-to-lis-office-environment)
show individual behaviors.

### Freezing the GRU encoder

The nine-seed confirmatory study trained full GRU-DQN for 15k episodes and
compared it with a policy trained jointly through 10k, then head-only through
15k. The paired return difference (frozen minus full) was −0.006 (95%
Student-t CI [−0.136, +0.124]); this does not satisfy the predeclared
non-inferiority margin of −0.05. Mean wall time fell from 2,331.979 s to
2,122.160 s, saving 209.820 s per run (9.00%; 95% CI 8.80%–9.19%).

![Confirmatory frozen-GRU comparison](artifacts/gru_freeze_ab/confirmatory_10k/comparison.png)

![Per-start paired frozen-minus-full return difference](artifacts/research_summary/grid_visualizations/freeze_frozen_minus_full_15k_by_start.png)

A seed-27 direct-stop run stopped at 12.5k after freezing at 10k and saved
24.35% wall time against a 15k full-GRU run. Its paired rollout return gain
was +0.097, but this is one seed. In the separate 25k extension, frozen GRU
scored 0.454 at 20k versus 0.414 for full GRU at 25k, then declined to 0.328
at frozen 25k. The evidence favors choosing a measured stop point over
continuing head-only training indefinitely, but does not establish
cross-seed quality preservation.

![Frozen-GRU return trajectory through 25k episodes](artifacts/gru_freeze_ab/convergence_25k/convergence.png)

The [freeze checkpoint return grids](artifacts/research_summary/grid_visualizations/index.md),
[policy-map index](artifacts/gru_freeze_ab/convergence_25k/policies/index.md),
and [full freeze protocol](GRU_FREEZE_EXPERIMENT.md) contain the remaining
visual and run-level detail.

### RSSM-inspired encoder feeding DQN

The seed-27 RSSM pilot uses four-token histories, a 16-unit deterministic
state, two four-class categorical latents, and a 2,979-parameter DQN model.
Its best checkpoint was at 3,750 episodes (0.3589 return, 454.4 s training
loop); return dropped to 0.1952 at 5,000 episodes. The best RSSM checkpoint
was below full-GRU 25k (0.4142), frozen-GRU 20k (0.4542), and noisy-input VI
(0.3989), but above frozen-GRU 25k (0.3275).

![Return versus training time for RSSM and GRU checkpoints](artifacts/rssm_dqn_pilot/return_vs_training_time.png)

![RSSM and GRU training-return trajectories](artifacts/rssm_dqn_pilot/training_return_trajectories.png)

The per-cell RSSM and matched-GRU return figures are in the [notebook-style
grid gallery](artifacts/research_summary/grid_visualizations/index.md).
Noise-averaged greedy action and expected-Q maps for all seven checkpoints
are in the [RSSM policy-map index](artifacts/rssm_dqn_pilot/policies/index.md).
This is one training seed, the implementation is RSSM-inspired rather than a
canonical RSSM, and the pilot used one eighth as many DQN updates per
environment step as the GRU baseline. It is useful feasibility evidence, not
an equal-update architecture comparison.

## Full report and data

- [Experiment 1 and office results, per-cell tables, and references](EXPERIMENTS.md)
- [Section One research checkpoint](SECTION_ONE_CHECKPOINT.md)
- [GRU-freezing design, timing, and seed-level results](GRU_FREEZE_EXPERIMENT.md)
- [RSSM design, hyperparameters, wall times, paired results, and limitations](RSSM_DQN_PILOT.md)
- [All 23 notebook-style grid visuals](artifacts/research_summary/grid_visualizations/index.md)

Machine-readable rollout reports and training manifests are linked from those
experiment documents. Confidence intervals over rollout scenarios describe
evaluation sampling; intervals across training seeds are identified
separately. Single-seed follow-ups are labeled exploratory throughout.
