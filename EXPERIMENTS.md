# History-4 POMDP experiments

The consolidated results and visual gallery are in the [Section One research
report](RESEARCH_RESULTS_REPORT.md); the [notebook-style return-grid
figures](artifacts/research_summary/grid_visualizations/index.md) use the same
`GridMDP.plot` visualizer as the notebooks.

## Question

Does a GRU-DQN using four recent observations outperform both a matched
feedforward DQN with no history and a conventional product-MDP value-iteration
policy when each is evaluated under partial observability? The GRU/MLP pair
isolates the contribution of observation history; noisy-input value iteration
measures performance against the regular MDP policy with noisy state indices.

The matched GRU versus no-GRU runs use the same environment, seeds, episode
budget, step budget, replay settings, and evaluation scenarios. The no-GRU
policy is a two-hidden-layer 64-unit MLP that receives only the current noisy
position and constant Rabin-pair index; the task automaton state is hidden from both
policies. Across ten matched training seeds, the GRU outperformed the MLP in
paired evaluation on every seed. The final ten-seed evaluation is complete.

A seed-27 comparison of full-GRU, frozen-GRU, and a compact RSSM-inspired
encoder feeding DQN is also complete. The best RSSM checkpoint reached 0.359
mean return after 7.57 minutes of training, below full-GRU 25k (0.414) and
frozen-GRU 20k (0.454), but above frozen-GRU 25k (0.328). It remained slightly
below the noisy-input MDP value-iteration baseline. This is a one-seed
feasibility result, and the RSSM used fewer optimizer updates per environment
step; details, paired intervals, training timings, and all grid maps are in
[the RSSM-DQN pilot report](RSSM_DQN_PILOT.md).

## Training protocol

The closest thesis experiment is Chapter 4, Section 4.5.1, Task 1 (Li, 2024),
which studies the same recurring `a`-or-`b` task while avoiding `c`. It trains
for 15,000 episodes of 600 steps, with a five-observation sequence, batch size
32, and target-network copies every 50 steps. This experiment uses the same
episode and step budget, batch size, and target-copy interval, with history
length four as requested.

The thesis uses a 10 by 10 grid, 0.9 intended-move probability, 0.1 sensor
error, discount 0.98, and an LSTM. This project keeps its existing 5 by 4 grid,
0.8 intended-move probability, 0.25 sensor error, discounts 0.99999/0.99, and
GRU architecture. It also retains the existing learning rate (1e-4), 64-unit
recurrent state, replay capacity (100,000), epsilon floor (0.1), epsilon decay
fraction (0.25), and epsilon action probability (0.2). Experience replay
optimization occurs every four environment steps in this implementation;
target weights are copied every 50 environment steps. The target interval is
measured in environment steps, not episodes.

The thesis's Task 1 gives the agent an automaton-state history. This run keeps
the repository's stricter partial-observation input: the true automaton state
is hidden, and the GRU receives position observations, previous actions, and
its own remembered epsilon target.

Ten 15k GRU runs (seeds 17–26), ten matched no-GRU runs, and value-iteration
evaluations for all ten GRU seeds have finished. GRU beat the no-GRU policy on
all ten matched seeds and noisy-input value iteration on nine. Four matched
30k, 45k, and 90k GRU runs (seeds 17–20) and their common-scenario evaluations
are also complete. At 90k, all four GRU policies beat noisy-input VI. Doubling
training from 45k to 90k did not produce a measurable average gain; see the
training-length analysis below. The office transfer experiment is complete as
well. A separate nine-seed training-cost ablation found that freezing the GRU
after 10k episodes saved about 9% wall time, but its confidence interval was
too wide to establish that final policy quality was preserved; see the final
section.
Reproduce the initial GRU runs with:

```sh
for seed in 17 18 19; do
  venv/bin/python train_pomdp.py --seed "$seed"
done
```

By default, each seed writes to its own `checkpoints_pomdp_history4_seed<N>`
directory so concurrent runs cannot overwrite each other's checkpoints.
Seeds 17 and 18 were first run before that seed-specific default was added; their
checkpoint and metadata files were later moved into the seed-specific directories.
Seed 19's retained model is the isolated rerun after the initial shared-directory
checkpoint-name collision.

## Baselines and evaluation

The primary no-GRU baseline is a feedforward MLP DQN trained for the same
15,000 episodes and 600 steps per episode as each history-4 GRU. It receives
the current noisy `(Rabin-pair index, row, column)` observation only. Both models hide
the true automaton state. The MLP receives neither observation history nor
previous actions. Evaluation pairs each matched training seed on common
rollout seeds, every traversable start square, 100 rollouts per square, and a
1,000-step horizon. The main statistic is the paired GRU-minus-MLP discounted
return difference, summarized across independent training seeds.

```sh
venv/bin/python train_pomdp.py --seed 17 --architecture mlp \
  --episodes 15000 --horizon 600 \
  --checkpoint-dir checkpoints_pomdp_mlp_seed17
venv/bin/python evaluate_pomdp_gru_vs_no_gru.py \
  --gru checkpoints_pomdp_history4_seed17/dqn_*.pt \
  --no-gru checkpoints_pomdp_mlp_seed17/dqn_*.pt \
  --runs 100 --horizon 1000 --seed 17 \
  --output artifacts/pomdp_history4_vs_feedforward_dqn.json
```

The secondary baseline computes value iteration on the exact product MDP, then
deploys that policy using only the noisy position reading. Its automaton state
is initialized at `q0`, tracks its own epsilon commitment, and is updated from
labels inferred at noisy positions. It treats each noisy position and inferred
automaton state as exact table indices; it does not maintain a belief over
hidden states. This is the tabular full-state policy evaluated under wrong
observations, rather than a tabular POMDP solver.

A full-state value-iteration oracle is also reported as context. It receives
the simulator's true product state and is not the primary comparison.

Each policy is evaluated on every traversable start square with the same
rollout seeds, 100 rollouts per square, and a 1,000-step horizon. The report
includes discounted returns, per-start results, paired GRU-minus-baseline
differences, and 95% confidence intervals. The report also aggregates
checkpoint means across training seeds with small-sample Student-t intervals.
Results are summarized both over all starts and over starts not labeled `c`.

```sh
venv/bin/python evaluate_pomdp_vs_value_iteration.py \
  checkpoints_pomdp_history4_seed17/dqn_*.pt \
  checkpoints_pomdp_history4_seed18/dqn_*.pt \
  checkpoints_pomdp_history4_seed19/dqn_*.pt \
  --runs 100 --horizon 1000 --seed 17 \
  --output artifacts/pomdp_history4_vs_mdp_value_iteration.json
```

This repository's environment and reward scales differ from the thesis, so
the thesis is used as a training-protocol reference rather than as a source of
expected return values.

## Results

Training returns improved from the first to the last 1,000 episodes for all
ten completed GRU seeds. Durations are wall-clock training time.

| Training seed | Mean return | First 1,000 episodes | Last 1,000 episodes | Duration |
|---:|---:|---:|---:|---:|
| 17 | 0.226 | 0.121 | 0.234 | 72.8 min |
| 18 | 0.221 | 0.124 | 0.253 | 76.5 min |
| 19 | 0.203 | 0.121 | 0.210 | 72.9 min |
| 20 | 0.213 | 0.107 | 0.209 | 90.4 min |
| 21 | 0.208 | 0.127 | 0.205 | 90.3 min |
| 22 | 0.214 | 0.130 | 0.209 | 90.6 min |
| 23 | 0.220 | 0.134 | 0.248 | 92.9 min |
| 24 | 0.240 | 0.130 | 0.270 | 93.5 min |
| 25 | 0.215 | 0.125 | 0.268 | 95.2 min |
| 26 | 0.219 | 0.125 | 0.247 | 95.8 min |

Notebook-style policy maps for each completed GRU checkpoint are saved with
the corresponding training artifacts:

| Training seed | 15k GRU map | No-GRU map | 30k map | 45k map | 90k map |
|---:|:---|:---|:---|:---|:---|
| 17 | [history-4 GRU](artifacts/pomdp_policy_maps/seed17_gru.png) | [MLP](artifacts/pomdp_policy_maps/seed17_mlp.png) | [30k run](artifacts/pomdp_policy_maps/seed17_gru_30k.png) | [45k run](artifacts/pomdp_policy_maps/seed17_gru_45k.png) | [90k run](artifacts/pomdp_policy_maps/seed17_gru_90k.png) |
| 18 | [history-4 GRU](artifacts/pomdp_policy_maps/seed18_gru.png) | [MLP](artifacts/pomdp_policy_maps/seed18_mlp.png) | [30k run](artifacts/pomdp_policy_maps/seed18_gru_30k.png) | [45k run](artifacts/pomdp_policy_maps/seed18_gru_45k.png) | [90k run](artifacts/pomdp_policy_maps/seed18_gru_90k.png) |
| 19 | [history-4 GRU](artifacts/pomdp_policy_maps/seed19_gru.png) | [MLP](artifacts/pomdp_policy_maps/seed19_mlp.png) | [30k run](artifacts/pomdp_policy_maps/seed19_gru_30k.png) | [45k run](artifacts/pomdp_policy_maps/seed19_gru_45k.png) | [90k run](artifacts/pomdp_policy_maps/seed19_gru_90k.png) |
| 20 | [history-4 GRU](artifacts/pomdp_policy_maps/seed20_gru.png) | [MLP](artifacts/pomdp_policy_maps/seed20_mlp.png) | [30k run](artifacts/pomdp_policy_maps/seed20_gru_30k.png) | [45k run](artifacts/pomdp_policy_maps/seed20_gru_45k.png) | [90k run](artifacts/pomdp_policy_maps/seed20_gru_90k.png) |
| 21 | [history-4 GRU](artifacts/pomdp_policy_maps/seed21_gru.png) | [MLP](artifacts/pomdp_policy_maps/seed21_mlp.png) | — | — | — |
| 22 | [history-4 GRU](artifacts/pomdp_policy_maps/seed22_gru.png) | [MLP](artifacts/pomdp_policy_maps/seed22_mlp.png) | — | — | — |
| 23 | [history-4 GRU](artifacts/pomdp_policy_maps/seed23_gru.png) | [MLP](artifacts/pomdp_policy_maps/seed23_mlp.png) | — | — | — |
| 24 | [history-4 GRU](artifacts/pomdp_policy_maps/seed24_gru.png) | [MLP](artifacts/pomdp_policy_maps/seed24_mlp.png) | — | — | — |
| 25 | [history-4 GRU](artifacts/pomdp_policy_maps/seed25_gru.png) | [MLP](artifacts/pomdp_policy_maps/seed25_mlp.png) | — | — | — |
| 26 | [history-4 GRU](artifacts/pomdp_policy_maps/seed26_gru.png) | [MLP](artifacts/pomdp_policy_maps/seed26_mlp.png) | — | — | — |

The ten completed no-history MLP runs have lower exploration-phase training
returns. These are training diagnostics; the paired greedy evaluation below
is the performance comparison.

| Training seed | Mean return | First 1,000 episodes | Last 1,000 episodes | Duration |
|---:|---:|---:|---:|---:|
| 17 | 0.089 | 0.104 | 0.087 | 69.6 min |
| 18 | 0.092 | 0.115 | 0.090 | 69.5 min |
| 19 | 0.089 | 0.105 | 0.090 | 69.6 min |
| 20 | 0.089 | 0.105 | 0.075 | 69.3 min |
| 21 | 0.090 | 0.097 | 0.094 | 69.3 min |
| 22 | 0.086 | 0.116 | 0.093 | 69.5 min |
| 23 | 0.092 | 0.123 | 0.092 | 69.2 min |
| 24 | 0.093 | 0.129 | 0.102 | 69.2 min |
| 25 | 0.090 | 0.111 | 0.094 | 48.5 min |
| 26 | 0.086 | 0.105 | 0.090 | 48.2 min |

### Direct GRU versus no-history MLP

Ten matched training seeds were evaluated greedily on the same 1,900 rollout
scenarios per policy (100 paired rollouts from each of 19 traversable starts,
with a 1,000-step horizon). Every GRU seed had higher mean discounted return
than its matched MLP seed.

| Training seed | GRU return | No-history MLP return | Paired difference, GRU − MLP (95% rollout CI) |
|---:|---:|---:|:---|
| 17 | 0.572 | 0.105 | +0.466 (+0.443, +0.490) |
| 18 | 0.435 | 0.077 | +0.358 (+0.335, +0.380) |
| 19 | 0.461 | 0.017 | +0.444 (+0.422, +0.467) |
| 20 | 0.461 | 0.105 | +0.356 (+0.334, +0.377) |
| 21 | 0.676 | 0.125 | +0.551 (+0.528, +0.574) |
| 22 | 0.607 | 0.109 | +0.497 (+0.474, +0.520) |
| 23 | 0.262 | 0.067 | +0.195 (+0.176, +0.213) |
| 24 | 0.511 | 0.099 | +0.412 (+0.390, +0.435) |
| 25 | 0.645 | 0.063 | +0.582 (+0.560, +0.604) |
| 26 | 0.504 | 0.094 | +0.409 (+0.385, +0.434) |

Across the ten independent training seeds, GRU mean return was 0.513
(95% Student-t CI 0.427–0.600), versus 0.086 for the MLP (0.064–0.109).
The mean paired improvement was +0.427 (95% CI +0.348 to +0.506). On starts
not labeled `c`, it was +0.541 (+0.441 to +0.641). This is direct evidence
that the history-4 GRU outperformed the matched no-history policy in this
partially observable environment.

![Paired GRU versus no-history MLP results on the task grid](artifacts/pomdp_policy_maps/gru_vs_mlp_10seeds.png)

![Matched GRU and MLP training returns](artifacts/pomdp_policy_maps/training_returns_10seeds.png)

The paired-evaluation data, including per-start returns and rollout samples,
are in `artifacts/pomdp_history4_vs_feedforward_dqn_10seeds.json`.

### Training progress and convergence

The raw training return is noisy because success is sparse. Epsilon reaches
0.1 after episode 3,750, and the separate 0.2 probability of taking an
available automaton transition remains active throughout training. The plot
shows a 1,000-episode moving average; the table groups saved per-episode
returns into disjoint 3,000-episode blocks. These exploration-heavy training
returns are not the same measure as the final greedy-policy evaluation.

The matched ten-seed GRU and MLP training-return plot above shows exploration-phase returns, not greedy policy quality.

| Episodes | Seed 17 | Seed 18 | Seed 19 | Seed 20 | Seed 21 | Seed 22 | Seed 23 | Seed 24 | Seed 25 | Seed 26 | Mean across 10 seeds |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1–3,000 | 0.159 | 0.166 | 0.153 | 0.149 | 0.142 | 0.162 | 0.154 | 0.155 | 0.151 | 0.157 | 0.155 |
| 3,001–6,000 | 0.245 | 0.235 | 0.222 | 0.238 | 0.210 | 0.255 | 0.227 | 0.249 | 0.228 | 0.235 | 0.234 |
| 6,001–9,000 | 0.254 | 0.229 | 0.222 | 0.216 | 0.226 | 0.223 | 0.226 | 0.255 | 0.216 | 0.218 | 0.229 |
| 9,001–12,000 | 0.239 | 0.228 | 0.208 | 0.230 | 0.232 | 0.217 | 0.251 | 0.276 | 0.228 | 0.240 | 0.235 |
| 12,001–15,000 | 0.233 | 0.247 | 0.210 | 0.233 | 0.231 | 0.213 | 0.239 | 0.264 | 0.254 | 0.245 | 0.237 |

The ten-seed training-return mean rises from 0.155 in episodes 1–3,000 to
about 0.23–0.24 thereafter. This exploration-phase plateau does not establish
policy convergence. Four matched GRU policies at 15k, 30k, and 45k episodes
were evaluated on the same 1,900 rollout scenarios per training seed (100
rollouts from each of 19 traversable starts, 1,000-step horizon). Each length
was trained from scratch with the same seed and epsilon-floor episode (3,750);
stored episode-return histories match exactly through every shared prefix.

| Training seed | 15k mean | 30k mean | 30k − 15k (95% rollout CI) | 45k mean | 45k − 30k (95% rollout CI) |
|---:|---:|---:|---:|---:|---:|
| 17 | 0.57160 | 0.59818 | +0.02658 (+0.00587, +0.04728) | 0.55093 | −0.04725 (−0.06703, −0.02747) |
| 18 | 0.43512 | 0.58435 | +0.14923 (+0.13098, +0.16748) | 0.60669 | +0.02234 (+0.00306, +0.04161) |
| 19 | 0.46097 | 0.52117 | +0.06020 (+0.04223, +0.07818) | 0.60859 | +0.08742 (+0.06851, +0.10633) |
| 20 | 0.46093 | 0.46138 | +0.00045 (−0.01238, +0.01328) | 0.57268 | +0.11130 (+0.09343, +0.12917) |
| **Mean across seeds** | **0.48215** | **0.54127** | **+0.05911 (−0.04408, +0.16231)** | **0.58472** | **+0.04345 (−0.06983, +0.15673)** |

The paired intervals for each individual seed quantify rollout sampling error;
the bold change intervals are Student-t intervals over four independent
training-seed differences. Three of four seeds improved from 30k to 45k, while
seed 17 declined. The mean increase was +0.04345, but its across-seed 95% CI
includes zero. Thus 45k is not a demonstrated plateau, and longer training has
not established a reliable average gain. The 45k mean (0.58472, across-seed
95% CI 0.54029–0.62915) exceeds the 15k mean (0.48215), but the paired 15k to
45k change is heterogeneous and inconclusive (mean +0.10257; 95% CI
−0.03387–+0.23900). Through 45k, more matched seeds or later greedy checkpoints
were needed to estimate expected improvement and convergence; the 90k results
below add a later comparison but leave the four-seed interval wide. Training
returns remain a poor stopping metric because they are exploration-phase
rewards.

![Paired 15k, 30k, and 45k policy returns across four training seeds](artifacts/pomdp_policy_maps/training_length_15k_30k_45k_seeds17_20.png)

The added runs use the same 15k protocol for seeds 20–26. The 30k seed-17 run
sets epsilon decay to 0.125 so its floor is still reached at episode 3,750,
matching the original run's schedule; this compares a longer training run
with its matched 15k checkpoint. The 30k job starts from the same random seed
and is trained from scratch; its first 15k returns match the 15k run exactly,
so it is not a resumed optimizer/replay checkpoint.

```sh
for seed in 20 21 22 23 24; do
  venv/bin/python train_pomdp.py --seed "$seed"
done
for seed in 25 26; do
  venv/bin/python train_pomdp.py --seed "$seed"
done
venv/bin/python train_pomdp.py --seed 17 --episodes 30000 \
  --epsilon-decay-fraction 0.125 \
  --checkpoint-dir checkpoints_pomdp_history4_seed17_30k
for seed in 18 19 20; do
  venv/bin/python train_pomdp.py --seed "$seed" --episodes 30000 \
    --epsilon-decay-fraction 0.125 \
    --checkpoint-dir "checkpoints_pomdp_history4_seed${seed}_30k"
done
venv/bin/python evaluate_pomdp_vs_value_iteration.py \
  checkpoints_pomdp_history4_seed17/dqn_*.pt \
  checkpoints_pomdp_history4_seed18/dqn_*.pt \
  checkpoints_pomdp_history4_seed19/dqn_*.pt \
  checkpoints_pomdp_history4_seed20/dqn_*.pt \
  checkpoints_pomdp_history4_seed17_30k/dqn_*.pt \
  checkpoints_pomdp_history4_seed18_30k/dqn_*.pt \
  checkpoints_pomdp_history4_seed19_30k/dqn_*.pt \
  checkpoints_pomdp_history4_seed20_30k/dqn_*.pt \
  --runs 100 --horizon 1000 --seed 17 --no-oracle \
  --include-rollout-samples \
  --output artifacts/pomdp_history4_length_eval_seeds17_20_timed.json
venv/bin/python analyze_pomdp_training_length.py \
  artifacts/pomdp_history4_length_eval_seeds17_20_timed.json \
  --output artifacts/pomdp_history4_length_comparison_seeds17_20.json \
  --plot artifacts/pomdp_policy_maps/training_length_15k_vs_30k_seeds17_20.png
```

The 30k training jobs took 9,524.348 seconds for seed 17, 6,619.517 for seed
18, 6,669.234 for seed 19, and 6,663.425 for seed 20. Their wall time per
episode was 0.3175, 0.2207, 0.2223, and 0.2221 seconds, respectively. Seeds
18–20 ran concurrently, so these durations include shared CPU load. Exact UTC
start and finish times are recorded in checkpoint metadata. Timed greedy
evaluation of all eight checkpoints took 1,292.110 seconds (21 min 32 s), from
2026-11-04 10:59:53.905 UTC to 11:21:26.014 UTC; individual checkpoint
evaluations took 161.077–162.086 seconds. Exact per-checkpoint timestamps and
durations, all rollout samples, and the paired summary are retained in
`artifacts/pomdp_history4_length_eval_seeds17_20_timed.json` and
`artifacts/pomdp_history4_length_comparison_seeds17_20.json`.

The later 45k training and three-length evaluation can be reproduced with:

```sh
for seed in 17 18 19 20; do
  venv/bin/python train_pomdp.py --seed "$seed" --episodes 45000 \
    --epsilon-decay-fraction 0.08333333333333333 \
    --checkpoint-dir "checkpoints_pomdp_history4_seed${seed}_45k"
done
venv/bin/python evaluate_pomdp_vs_value_iteration.py \
  checkpoints_pomdp_history4_seed17/dqn_*.pt \
  checkpoints_pomdp_history4_seed18/dqn_*.pt \
  checkpoints_pomdp_history4_seed19/dqn_*.pt \
  checkpoints_pomdp_history4_seed20/dqn_*.pt \
  checkpoints_pomdp_history4_seed17_30k/dqn_*.pt \
  checkpoints_pomdp_history4_seed18_30k/dqn_*.pt \
  checkpoints_pomdp_history4_seed19_30k/dqn_*.pt \
  checkpoints_pomdp_history4_seed20_30k/dqn_*.pt \
  checkpoints_pomdp_history4_seed17_45k/dqn_*.pt \
  checkpoints_pomdp_history4_seed18_45k/dqn_*.pt \
  checkpoints_pomdp_history4_seed19_45k/dqn_*.pt \
  checkpoints_pomdp_history4_seed20_45k/dqn_*.pt \
  --runs 100 --horizon 1000 --seed 17 --no-oracle \
  --include-rollout-samples \
  --output artifacts/pomdp_history4_length_eval_seeds17_20_15k_30k_45k_timed.json
venv/bin/python analyze_pomdp_training_length.py \
  artifacts/pomdp_history4_length_eval_seeds17_20_15k_30k_45k_timed.json \
  --output artifacts/pomdp_history4_length_comparison_seeds17_20_15k_30k_45k.json \
  --plot artifacts/pomdp_policy_maps/training_length_15k_30k_45k_seeds17_20.png
```

The four 45k continuations were trained from scratch with seeds 17–20 and an
epsilon-decay fraction of 0.0833333, keeping the epsilon floor at episode
3,750 as in the 15k runs. Each used 45,000 episodes × 600 steps (27,000,000
environment steps), history length 4, batch 32, replay capacity 100,000,
learning rate 1e-4, replay optimization every four environment steps, target
copy every 50 steps, and the existing 0.2 automaton-action probability. This
is 6,750,000 replay-optimizer calls (the initial calls skip until the 32-item
batch is available) and exactly 540,000 target copies per run. Their first
15,000 and 30,000 stored episode returns match the corresponding shorter runs
element by element. All four jobs started within 2 milliseconds of one another,
so each elapsed time includes shared CPU contention.

| Seed | Started (UTC) | Finished (UTC) | Wall seconds | Seconds/episode | Mean training return | First 1,000 | Last 1,000 |
|---:|:---|:---|---:|---:|---:|---:|---:|
| 17 | 2026-11-04T11:24:00.035432+00:00 | 2026-11-04T14:43:40.556937+00:00 | 11,980.494 | 0.26623 | 0.226100 | 0.120838 | 0.219516 |
| 18 | 2026-11-04T11:24:00.037096+00:00 | 2026-11-04T14:43:54.542257+00:00 | 11,994.478 | 0.26654 | 0.227030 | 0.123850 | 0.227493 |
| 19 | 2026-11-04T11:24:00.035475+00:00 | 2026-11-04T14:43:15.967577+00:00 | 11,955.905 | 0.26569 | 0.208888 | 0.120818 | 0.222486 |
| 20 | 2026-11-04T11:24:00.035419+00:00 | 2026-11-04T14:41:04.096890+00:00 | 11,824.035 | 0.26276 | 0.227825 | 0.106902 | 0.237462 |

The exact timing and training-return records are in each 45k checkpoint's JSON
sidecar. All 12 matched checkpoints were evaluated on evaluation seed 17 and
the same 1,900 rollout scenarios per checkpoint. The batch took 3,162.126
seconds (52 min 42 s), from 2026-11-04T14:45:27.533Z to
2026-11-04T15:38:09.658Z. Per-checkpoint time ranged from 162.832 to 315.665
seconds; exact start, finish, and elapsed times are in the timed JSON report.
The four 45k policies averaged 0.58472 return versus 0.39890 for noisy-input
value iteration. Their mean paired advantage was +0.18582 (95% Student-t CI
across training seeds +0.14140 to +0.23025; n=4).

### 90k extension and convergence check

These matched GRU runs use 90,000 episodes × 600 steps (54,000,000
environment steps per seed), the same history, optimizer, target-update,
replay, and automaton-action settings as 15k/45k, and an epsilon floor at
episode 3,750. All four 90k episode-return histories match their 45k histories
exactly over the first 45,000 episodes.

Seeds 17–20 are the four seeds already used in the matched 15k/30k/45k
length comparison. Reusing them makes the 45k-to-90k comparison paired by
training seed and keeps seed-to-seed variation from obscuring the effect of
training length. Four seeds are a practical extension of the existing cohort,
not a claim that between-seed uncertainty is small; the confidence interval
will remain imprecise at this sample size.

Training-return means for the four completed 90k runs are:

| Seed | Episodes 45,001–60,000 | 60,001–75,000 | 75,001–90,000 |
|---:|---:|---:|---:|
| 17 | 0.21412 | 0.21385 | 0.21106 |
| 18 | 0.22582 | 0.19649 | 0.21106 |
| 19 | 0.19943 | 0.20634 | 0.21633 |
| 20 | 0.22017 | 0.22696 | 0.21878 |
| **Mean of four seeds** | **0.21489** | **0.21091** | **0.21431** |

These exploration-phase returns show no consistent upward trend after episode
45,000, but they cannot establish greedy-policy convergence: epsilon remains at
0.1, the automaton-transition action probability remains 0.2, and success is
sparse. Greedy policies were therefore evaluated on matched rollout scenarios.

| Seed | Started (UTC) | Finished (UTC) | Wall seconds | Seconds/episode | Mean training return | First 1,000 | Last 1,000 | Prefix check |
|---:|:---|:---|---:|---:|---:|---:|---:|:---:|
| 17 | 2026-11-04T15:41:03.597639+00:00 | 2026-11-04T21:49:59.105539+00:00 | 22,135.490 | 0.2460 | 0.219556 | 0.120838 | 0.214507 | Exact to 45k |
| 18 | 2026-11-04T21:50:01.177669+00:00 | 2026-11-05T01:49:40.347186+00:00 | 14,379.168 | 0.1598 | 0.219078 | 0.123850 | 0.230459 | Exact to 45k |
| 19 | 2026-11-05T01:49:42.353573+00:00 | 2026-11-05T05:49:51.432623+00:00 | 14,409.009 | 0.1601 | 0.208128 | 0.120818 | 0.230448 | Exact to 45k |
| 20 | 2026-11-05T05:49:53.407544+00:00 | 2026-11-05T09:49:15.862328+00:00 | 14,362.427 | 0.1596 | 0.224898 | 0.106902 | 0.214486 | Exact to 45k |

Seed 17 took 6 h 8 min 55 s; seed 18 took 3 h 59 min 39 s; seed 19 took
4 h 00 min 09 s; seed 20 took 3 h 59 min 22 s. The exact histories and timing
metadata are retained in the seed-specific checkpoint sidecars. All four
90k policy maps are linked above.

The four final 90k policies were evaluated with evaluation seed 17, 100
rollouts from each of 19 traversable starts, and a 1,000-step horizon. The
batch ran from 2026-11-05T09:50:39.112Z to 10:01:28.541Z (649.426 seconds).
Individual checkpoint evaluations took 161.967–162.707 seconds. The report
contains per-start results and paired rollout samples.

| Training seed | 90k GRU return | Noisy-input VI return | GRU − VI (95% rollout CI) | 45k → 90k change (95% rollout CI) |
|---:|---:|---:|:---|:---|
| 17 | 0.52526 | 0.39890 | +0.12636 (+0.09982, +0.15290) | −0.02567 (−0.04822, −0.00312) |
| 18 | 0.62725 | 0.39890 | +0.22835 (+0.20587, +0.25083) | +0.02056 (+0.00143, +0.03968) |
| 19 | 0.45994 | 0.39890 | +0.06104 (+0.03741, +0.08468) | −0.14865 (−0.16818, −0.12912) |
| 20 | 0.59452 | 0.39890 | +0.19562 (+0.17181, +0.21943) | +0.02184 (+0.00071, +0.04298) |
| **Mean across four seeds** | **0.55174** | **0.39890** | **+0.15284 (+0.03428, +0.27140)** | **−0.03298 (−0.16061, +0.09465)** |

The rollout intervals in the first four rows quantify sampling uncertainty for
each trained policy. The last column's per-seed intervals are also rollout
intervals; its mean-across-seeds interval is a Student-t interval over four
independent training-seed differences. The GRU beats noisy-input VI on all four
90k seeds, and the across-seed paired interval is above zero. Thus the main
claim is supported at both budgets: at 15k the GRU beat noisy-input VI in 9/10
seeds (+0.114, 95% CI +0.028 to +0.201), and at 90k it beat it in 4/4 seeds
(+0.153, 95% CI +0.034 to +0.271). This comparison is specifically against
ordinary product-MDP value iteration deployed with noisy state indices; it is
not a comparison against a belief-state POMDP solver.

The training-length analysis shows no measurable average gain from doubling
45k to 90k: the mean paired change is −0.03298, with a wide 95% interval
[−0.16061, +0.09465]. The interval still allows an improvement of about 16%
of the 45k mean, as well as a larger decline. Two seeds improve slightly and
two decline; seed 19 has the largest decline. Therefore these data do not
demonstrate convergence, but they also show no consistent improvement from
more episodes. More episodes are not needed to establish the
GRU-versus-noisy-VI result. If a stronger convergence claim is required, the
next useful experiment is more independent training seeds and greedy
evaluations at intermediate checkpoints, rather than simply extending the
same four runs.

![Matched 15k, 30k, 45k, and 90k policy returns and paired changes](artifacts/pomdp_policy_maps/training_length_15k_30k_45k_90k_seeds17_20.png)

Machine-readable data: [90k versus VI evaluation](artifacts/pomdp_history4_eval_90k_seeds17_20_timed.json), [all-length matched rollouts](artifacts/pomdp_history4_length_eval_seeds17_20_15k_30k_45k_90k_timed.json), and [paired training-length analysis](artifacts/pomdp_history4_length_comparison_seeds17_20_15k_30k_45k_90k.json).

All ten history-4 seeds have now been evaluated against the noisy-input MDP
value-iteration baseline. Nine had a higher mean discounted return; seed 23
was lower.

| Training seed | GRU-DQN mean | VI on noisy observation | Paired difference (95% rollout CI) | Non-`c` paired difference (95% rollout CI) |
|---:|---:|---:|---:|---:|
| 17 | 0.572 | 0.399 | +0.173 (0.149, 0.196) | +0.219 (0.189, 0.248) |
| 18 | 0.435 | 0.399 | +0.036 (0.014, 0.059) | +0.046 (0.017, 0.075) |
| 19 | 0.461 | 0.399 | +0.062 (0.037, 0.087) | +0.079 (0.047, 0.110) |
| 20 | 0.461 | 0.399 | +0.062 (0.039, 0.085) | +0.079 (0.049, 0.108) |
| 21 | 0.676 | 0.399 | +0.277 (0.254, 0.300) | +0.351 (0.323, 0.379) |
| 22 | 0.607 | 0.399 | +0.208 (0.184, 0.232) | +0.263 (0.233, 0.293) |
| 23 | 0.262 | 0.399 | −0.137 (−0.162, −0.112) | −0.173 (−0.205, −0.142) |
| 24 | 0.511 | 0.399 | +0.112 (0.089, 0.136) | +0.142 (0.113, 0.172) |
| 25 | 0.645 | 0.399 | +0.246 (+0.224, +0.269) | +0.312 (+0.284, +0.340) |
| 26 | 0.504 | 0.399 | +0.105 (+0.081, +0.129) | +0.133 (+0.102, +0.163) |

Across ten independent training seeds, the GRU mean return was 0.513
(95% Student-t CI 0.427–0.600), versus 0.399 for noisy-input VI. The mean
paired difference was +0.114 (95% CI +0.028 to +0.201); on starts not labeled
`c`, it was +0.145 (+0.036 to +0.254). Both cross-seed intervals exclude zero.
Nine seeds beat the baseline on paired rollouts; seed 23 remains a clear
counterexample. The earlier eight-seed estimate was positive but inconclusive
(+0.099, 95% CI −0.006 to +0.205); the two additional independent seeds
narrowed the interval enough to support a positive average difference.

![Paired GRU gains over MDP value iteration indexed by noisy observations](artifacts/pomdp_policy_maps/gru_vs_noisy_mdp_vi_10seeds.png)

The full-state value-iteration policy averaged 0.765, falling to 0.399 when
indexed using noisy observations (a 0.366 drop). The history-4 GRU averaged
0.513 in the partially observable environment, 0.114 above the noisy-input
baseline; descriptively, that closes about 31% of the mean-return loss from
using noisy state indices. The GRU mean remains 0.251 below the full-state
reference, so this is an advantage over conventional MDP VI under the
specified observation noise, not an advantage over tabular VI when the correct
state is available or over an optimal POMDP solver.

Returns near 1 indicate that the policy reaches and sustains accepting
behavior; returns near 0 indicate that it fails the task, often by entering a
rejecting `c` state. Intermediate values reflect the chance and timing of
reaching accepting behavior under transition and observation noise.

### Values arranged on the problem grid

All tables below use the problem's 5-row by 4-column layout: row 0 is at the
top, and columns run left to right. This reference table shows the cell types
and propositions so each value table can be read at the matching coordinate.
`·` is an ordinary unlabeled cell; `T:x` is a trap labeled `x`.

| Row | Col 0 | Col 1 | Col 2 | Col 3 |
|---:|:---:|:---:|:---:|:---:|
| 0 | · | · | c | · |
| 1 | · | · | a | T:b |
| 2 | B | · | c | · |
| 3 | T:b | · | T:a | · |
| 4 | · | c | · | c |

The first value table gives exact value iteration's value for each initial
product state `(Rabin-pair index 0, q=q0, row, column)`, with access to the
true product state (`--vi-threshold 1e-8`). The cell labels are in the
reference grid above.

| Row | Col 0 | Col 1 | Col 2 | Col 3 |
|---:|---:|---:|---:|---:|
| 0 | 0.998 | 0.998 | 0.000 | 1.000 |
| 1 | 0.998 | 0.998 | 0.800 | 1.000 |
| 2 | B | 0.998 | 0.000 | 1.000 |
| 3 | 1.000 | 0.999 | 1.000 | 1.000 |
| 4 | 1.000 | 0.000 | 0.800 | 0.000 |

The next table shows empirical 1,000-step discounted returns from each start
square, averaged over ten training seeds. Each open cell is `GRU-DQN / VI`
when VI is deployed using noisy observations. These are estimates of each
policy's return from that initial product state under the noisy environment;
they are not neural-network state values. The GRU takes a history as input and
has action values rather than a separately learned scalar state-value
function. The VI baseline is the same in each training-seed run because the
evaluation scenarios are paired.

| Row | Col 0 | Col 1 | Col 2 | Col 3 |
|---:|---:|---:|---:|---:|
| 0 | 0.464 / 0.470 | 0.537 / 0.330 | 0.000 / 0.000 | 0.870 / 0.130 |
| 1 | 0.480 / 0.360 | 0.489 / 0.370 | 0.700 / 0.600 | 1.000 / 1.000 |
| 2 | B | 0.422 / 0.350 | 0.000 / 0.000 | 0.810 / 0.440 |
| 3 | 1.000 / 1.000 | 0.670 / 0.300 | 1.000 / 0.930 | 0.840 / 0.470 |
| 4 | 0.890 / 0.260 | 0.000 / 0.000 | 0.690 / 0.570 | 0.000 / 0.000 |

Mean paired return difference by start square (`GRU-DQN − VI`):

| Row | Col 0 | Col 1 | Col 2 | Col 3 |
|---:|---:|---:|---:|---:|
| 0 | −0.006 | +0.207 | +0.000 | +0.740 |
| 1 | +0.120 | +0.119 | +0.100 | +0.000 |
| 2 | B | +0.072 | +0.000 | +0.370 |
| 3 | +0.000 | +0.370 | +0.070 | +0.370 |
| 4 | +0.630 | +0.000 | +0.120 | +0.000 |

The per-cell entries average across ten seeds; some cells remain at or below
VI even though the overall paired difference is positive. These cell means
are point estimates without per-cell across-seed intervals. Exact per-seed
cell values and rollout intervals remain in the JSON report.

Per-run and per-start values are in
`artifacts/pomdp_history4_vs_mdp_value_iteration_10seeds.json`.

The seed-17 30k run is a separate paired check against its matched 15k
checkpoint on the same 1,900 evaluation scenarios. The next grid gives
15k return → 30k return (paired difference) from each start. Its aggregate
paired difference was +0.027 (95% rollout CI +0.006 to +0.047); several
individual starts decreased, so this single-seed result does not show uniform
improvement across the map.

| Row | Col 0 | Col 1 | Col 2 | Col 3 |
|---:|:---:|:---:|:---:|:---:|
| 0 | 0.464 → 0.619 (+0.155) | 0.537 → 0.579 (+0.043) | 0.000 → 0.000 (+0.000) | 0.870 → 0.930 (+0.060) |
| 1 | 0.480 → 0.589 (+0.109) | 0.489 → 0.659 (+0.170) | 0.700 → 0.460 (−0.240) | 1.000 → 1.000 (+0.000) |
| 2 | B | 0.422 → 0.730 (+0.308) | 0.000 → 0.000 (+0.000) | 0.810 → 0.700 (−0.110) |
| 3 | 1.000 → 1.000 (+0.000) | 0.670 → 0.860 (+0.190) | 1.000 → 1.000 (+0.000) | 0.840 → 0.590 (−0.250) |
| 4 | 0.890 → 0.870 (−0.020) | 0.000 → 0.000 (+0.000) | 0.690 → 0.780 (+0.090) | 0.000 → 0.000 (+0.000) |

## Experiment 2: transfer to Li's office environment

This is a single-agent 4×4 office abstraction, using the state labels and
four-direction architectural signatures from Figure 11 and Li's public
supplementary office model. The repeated signatures are analytically grouped:
the `(wall, wall, door, window)` signature aliases the two west-facing window
offices, while `(wall, wall, door, wall)` aliases offices `b`, `c`, and the
printer room. The other eleven cells have unique signatures.

The supplementary model's full `s_property` table supplies these four-way
patterns. The separate `get_observation_prob` routine that samples one of the
four directions is not used.

The office readout uses the entire local four-direction signature. If a
signature identifies one cell, it returns that cell exactly. If the same
signature belongs to multiple cells, it returns each matching cell uniformly.
For true cell `s`, candidate readout `o` therefore has probability
`1 / |C(s)|` when `o` is in the same signature class `C(s)`, and zero otherwise.
This makes the ambiguous readout probabilities exact and repeatable; it does
not sample one sensor direction. The uniform alias rule is our explicit
observation abstraction over Li's office geometry; it is not a claim that the
thesis used this readout distribution.
Every unique-signature cell has readout probability 1.0; the two-cell window
alias has probability 0.5 per candidate, and the three-cell wall/door alias
has probability 1/3 per candidate. Other cells have probability zero for
each of those ambiguous states.

| Row | Col 0 | Col 1 | Col 2 | Col 3 |
|---:|:---|:---|:---|:---|
| 0 | W,W,D,Wi · `a` (alias ×2) | W,H,D,D | W,W,D,D | W,D,W,D |
| 1 | W,W,D,W · office `b` (alias ×3) | H,H,D,D | W,W,W,D · `S` | D,W,W,W · Sply |
| 2 | W,W,D,W · `c` (alias ×3) | H,H,W,D | W,W,D,W · Print (alias ×3) | W,H,W,D |
| 3 | W,W,D,Wi · office `d` (alias ×2) | H,W,H,D | W,W,H,H | H,W,W,H · start |

`W` = wall, `H` = hallway, `D` = door, and `Wi` = window; entries are the
north, south, east, and west features in that order. The goal is the thesis's
single-agent office Task 1: repeatedly visit `Print`, then `a` or `c`, while
avoiding storage `S`, represented by
`G F (p & F (a | c)) & G !s`.
Offices `b` and `d`, and `Sply`, are unlabeled for this task.

For context, the following table gives exact value iteration on the fully
observed office product MDP, initialized at automaton state `q0`, with the
thesis-aligned 0.98/0.99 discounts. This is an oracle reference with access to
the true location and automaton state; it is not the learned POMDP policy.

| Row | Col 0 | Col 1 | Col 2 | Col 3 |
|---:|---:|---:|---:|---:|
| 0 | 0.027 | 0.028 | 0.027 | 0.026 |
| 1 | 0.028 | 0.028 | 0.000 (`S`) | 0.026 |
| 2 | 0.030 (`c`) | 0.030 | 0.034 (`Print`) | 0.033 |
| 3 | 0.030 | 0.031 | 0.032 | 0.033 (start) |

The main office transfer group follows Li's closest model-free office budget
of 10,000 episodes × 300 steps, batch size 32, and target-network copy every
50 environment steps. It uses history length 4, a 64-unit GRU, learning rate
`1e-4`, replay capacity 100,000, and the same four-step replay update and
epsilon settings as Experiment 1. The motion probability is 0.9 for the
intended available direction; the remaining probability is shared across
other available directions, following the supplementary office model. The
discounts are 0.98 for nonaccepting states and 0.99 for accepting states,
matching the thesis's 0.98 discount while preserving the project's
acceptance-state convention. The true task automaton state is hidden from the
policy. Training starts at the hallway endpoint `(3,3)` shown in Figure 11.
Evaluation counts actual LDBA acceptance rewards and storage violations from
that thesis start, plus per-start returns and task metrics on the 4×4 grid.

The direct comparison in Experiment 1 is complete for ten matched seeds, with
the GRU outperforming the MLP on every seed. Five initial office diagnostics
(seeds 17–21) used the Experiment 1 nonaccepting discount (0.99999). Their
position-only `Print → a/c` counts overstated task completion because the LDBA
commits to one target branch; the corrected evaluator uses acceptance rewards.
All five checkpoints have been evaluated with 100 rollouts from the thesis
start. Only seed 21 completed two accepted cycles in a substantial fraction of
rollouts, showing high seed-to-seed variation under this diagnostic discount.

| Training seed | Mean LTL-acceptance cycles | P(≥1 safe cycle) | P(≥2 safe cycles) | No storage violation | Mean discounted return |
|---:|---:|---:|---:|---:|---:|
| 17 | 0.02 | 0.02 | 0.00 | 1.00 | 0.00020 |
| 18 | 0.00 | 0.00 | 0.00 | 0.99 | 0.00000 |
| 19 | 0.08 | 0.06 | 0.00 | 0.87 | 0.00080 |
| 20 | 0.26 | 0.23 | 0.00 | 0.93 | 0.00259 |
| 21 | 1.47 | 0.80 | 0.45 | 0.96 | 0.01454 |

The five diagnostic training runs used 10,000 episodes of 300 steps each.

| Training seed | Mean training return | First 1,000 | Last 1,000 | Wall time |
|---:|---:|---:|---:|---:|
| 17 | 0.002581 | 0.004868 | 0.001178 | 35.46 min |
| 18 | 0.002549 | 0.004290 | 0.002175 | 35.46 min |
| 19 | 0.003090 | 0.004769 | 0.002455 | 35.53 min |
| 20 | 0.003802 | 0.004897 | 0.003234 | 17.98 min |
| 21 | 0.003014 | 0.004976 | 0.002275 | 17.98 min |

The rates above are point estimates from 100 rollouts per training seed. These
five runs are retained as a diagnostic group and are not pooled with the
thesis-discounted transfer group below. Across seeds 17–21, the thesis-start
mean was 0.37 accepted cycles; the mean probability of at least two safe
cycles was 0.09 (95% Student-t CI −0.16 to 0.34), too uncertain to establish
reliable repeated completion.

![Office diagnostic LTL returns and cycle completion arranged on the office grid](artifacts/office_policy_maps/diagnostic_grid_5seeds.png)

To match the thesis discount and address the weak diagnostic performance, the
main office transfer group (seeds 22–28) uses 0.98/0.99 discounts and the same
10,000 × 300 training budget. Training, all-start evaluation, and the
acceptance-reward evaluator are complete for all seven seeds. Each seed was
evaluated on 100 rollouts from the thesis start; the evaluation report also
contains 100-rollout measurements from every cell of the 4×4 grid.

| Training seed | Mean accepted cycles | P(≥1 safe cycle) | P(≥2 safe cycles) | No storage violation | Mean discounted return |
|---:|---:|---:|---:|---:|---:|
| 22 | 1.73 | 0.51 | 0.35 | 0.56 | 0.00164 |
| 23 | 0.93 | 0.22 | 0.06 | 0.28 | 0.00177 |
| 24 | 0.39 | 0.20 | 0.08 | 0.60 | 0.00064 |
| 25 | 0.80 | 0.57 | 0.19 | 0.94 | 0.00009 |
| 26 | 0.45 | 0.31 | 0.07 | 0.83 | 0.00132 |
| 27 | 0.15 | 0.05 | 0.00 | 0.47 | 0.00023 |
| 28 | 0.24 | 0.17 | 0.05 | 0.73 | 0.00017 |

Across the seven training seeds, the mean number of accepted cycles was 0.67
(95% Student-t CI 0.16–1.18). The probability of at least one safe accepted
cycle averaged 0.29 (0.12–0.46), and the probability of at least two safe
accepted cycles averaged 0.114 (0.004–0.224). The mean probability of avoiding
storage averaged 0.63 (0.42–0.84). These across-seed intervals exclude zero
for the one-cycle and two-cycle measures, showing transfer beyond a single
successful checkpoint. Performance varies materially across seeds: the
per-seed probability of two safe cycles ranges from 0.00 to 0.35, and the
storage-avoidance rate ranges from 0.28 to 0.94. This supports feasibility on
the thesis office abstraction; it does not establish consistently high
success or safety.

The group-average state values and task metrics are plotted in the same 4×4
layout as the office problem. The three panels show mean discounted return,
mean LTL-acceptance cycles, and the probability of at least two safe cycles.
The corresponding full-state value-iteration oracle grid appears above.

| Training seed | Mean training return | First 1,000 | Last 1,000 | Wall time |
|---:|---:|---:|---:|---:|
| 22 | 0.000968 | 0.000820 | 0.001208 | 20.13 min |
| 23 | 0.001000 | 0.000894 | 0.001395 | 20.10 min |
| 24 | 0.000717 | 0.000927 | 0.000870 | 20.14 min |
| 25 | 0.001242 | 0.000845 | 0.001604 | 15.27 min |
| 26 | 0.000687 | 0.000806 | 0.000809 | 15.26 min |
| 27 | 0.000708 | 0.000884 | 0.000939 | 14.88 min |
| 28 | 0.000802 | 0.000805 | 0.000974 | 14.86 min |

![Thesis-discounted office training returns across seven seeds](artifacts/office_policy_maps/training_returns_gamma098_7seeds.png)

![Office discounted return, LTL cycles, and safe two-cycle probability on the 4×4 grid](artifacts/office_policy_maps/office_gamma098_grid_7seeds.png)


The following seven-seed mean values are tabulated at the same row and column coordinates as the office layout. Each open cell aggregates 700 evaluation rollouts (100 per seed); `S` is the storage cell, where the safety metrics are zero by definition.

Mean discounted return by office cell:

| Row | Col 0 | Col 1 | Col 2 | Col 3 |
|---:|---:|---:|---:|---:|
| 0 | 0.000142 | 0.000088 | 0.000096 | 0.000122 |
| 1 | 0.000082 | 0.000183 | S | 0.000107 |
| 2 | 0.000307 | 0.000198 | 0.004157 | 0.001688 |
| 3 | 0.000164 | 0.000311 | 0.000345 | 0.000837 |

Mean LTL-acceptance cycles by office cell:

| Row | Col 0 | Col 1 | Col 2 | Col 3 |
|---:|---:|---:|---:|---:|
| 0 | 0.50 | 0.38 | 0.44 | 0.51 |
| 1 | 0.47 | 0.48 | S | 0.50 |
| 2 | 0.56 | 0.49 | 1.20 | 0.78 |
| 3 | 0.59 | 0.54 | 0.58 | 0.67 |

Probability of at least two accepted cycles without storage violation:

| Row | Col 0 | Col 1 | Col 2 | Col 3 |
|---:|---:|---:|---:|---:|
| 0 | 0.09 | 0.07 | 0.08 | 0.11 |
| 1 | 0.09 | 0.09 | S | 0.10 |
| 2 | 0.10 | 0.10 | 0.22 | 0.13 |
| 3 | 0.10 | 0.10 | 0.11 | 0.11 |

The office setting is based on
[Li's dissertation](https://doi.org/10.25820/etd.007512) and the
[supplementary 4×4 office model](https://github.com/JunchaoLi001/LDGBA-Model_Checking/blob/main/Office%20Simulation/Office%20single%20obsv%20by%20SARSOP/csrl/pomdp.py) and [office task notebook](https://github.com/JunchaoLi001/LDGBA-Model_Checking/blob/main/Office%20Simulation/Office%20single%20obsv%20by%20SARSOP/generate%20product%20POMDP.ipynb).

The office training and evaluation commands are:

```sh
for seed in 17 18 19; do
  venv/bin/python train_office_pomdp.py --seed "$seed" \
    --discount-nonaccepting 0.99999 --discount-accepting 0.99
done
venv/bin/python evaluate_office_pomdp.py \
  checkpoints_office_history4_seed17/dqn_*.pt \
  checkpoints_office_history4_seed18/dqn_*.pt \
  checkpoints_office_history4_seed19/dqn_*.pt \
  --runs 100 --horizon 1000 \
  --output artifacts/office_pomdp_evaluation_seed17_19_ltl.json
```

The thesis-discounted transfer runs use separate checkpoint directories:

```sh
for seed in 22 23 24 25 26 27 28; do
  venv/bin/python train_office_pomdp.py --seed "$seed" \
    --episodes 10000 --horizon 300 \
    --discount-nonaccepting 0.98 --discount-accepting 0.99 \
    --checkpoint-dir "checkpoints_office_history4_seed${seed}_gamma098"
done
venv/bin/python evaluate_office_pomdp.py \
  checkpoints_office_history4_seed22_gamma098/dqn_*.pt \
  checkpoints_office_history4_seed23_gamma098/dqn_*.pt \
  checkpoints_office_history4_seed24_gamma098/dqn_*.pt \
  checkpoints_office_history4_seed25_gamma098/dqn_*.pt \
  checkpoints_office_history4_seed26_gamma098/dqn_*.pt \
  checkpoints_office_history4_seed27_gamma098/dqn_*.pt \
  checkpoints_office_history4_seed28_gamma098/dqn_*.pt \
  --runs 100 --horizon 1000 --seed 17 \
  --output artifacts/office_pomdp_evaluation.json
```

The office policies have action/value and rollout maps. The maps show
expected max Q under the analytic wall-signature aliases, the modal greedy
action, and a representative rollout from the thesis start.

| Training seed | Policy map | Thesis-start rollout |
|---:|:---|:---|
| 17 | [Q/action map](artifacts/office_policy_maps/seed17_policy.png) | [rollout](artifacts/office_policy_maps/seed17_policy_route.png) |
| 18 | [Q/action map](artifacts/office_policy_maps/seed18_policy.png) | [rollout](artifacts/office_policy_maps/seed18_policy_route.png) |
| 19 | [Q/action map](artifacts/office_policy_maps/seed19_policy.png) | [rollout](artifacts/office_policy_maps/seed19_policy_route.png) |
| 20 | [Q/action map](artifacts/office_policy_maps/seed20_policy.png) | [rollout](artifacts/office_policy_maps/seed20_policy_route.png) |
| 21 | [Q/action map](artifacts/office_policy_maps/seed21_policy.png) | [rollout](artifacts/office_policy_maps/seed21_policy_route.png) |
| 22 | [Q/action map](artifacts/office_policy_maps/seed22_gamma098_policy.png) | [rollout](artifacts/office_policy_maps/seed22_gamma098_policy_route.png) |
| 23 | [Q/action map](artifacts/office_policy_maps/seed23_gamma098_policy.png) | [rollout](artifacts/office_policy_maps/seed23_gamma098_policy_route.png) |
| 24 | [Q/action map](artifacts/office_policy_maps/seed24_gamma098_policy.png) | [rollout](artifacts/office_policy_maps/seed24_gamma098_policy_route.png) |
| 25 | [Q/action map](artifacts/office_policy_maps/seed25_gamma098_policy.png) | [rollout](artifacts/office_policy_maps/seed25_gamma098_policy_route.png) |
| 26 | [Q/action map](artifacts/office_policy_maps/seed26_gamma098_policy.png) | [rollout](artifacts/office_policy_maps/seed26_gamma098_policy_route.png) |
| 27 | [Q/action map](artifacts/office_policy_maps/seed27_gamma098_policy.png) | [rollout](artifacts/office_policy_maps/seed27_gamma098_policy_route.png) |
| 28 | [Q/action map](artifacts/office_policy_maps/seed28_gamma098_policy.png) | [rollout](artifacts/office_policy_maps/seed28_gamma098_policy_route.png) |

![Office diagnostic training returns](artifacts/office_policy_maps/training_returns_5seeds.png)

For a newly trained policy, generate the same maps with:

```sh
venv/bin/python plot_office_policy_checkpoint.py \
  checkpoints_office_history4_seed22_gamma098/dqn_*.pt \
  --output artifacts/office_policy_maps/seed22_gamma098.png \
  --evaluation-report artifacts/office_pomdp_evaluation.json
```


## Training-cost ablation: freezing the GRU

A separate paired cohort compared history-4 GRU-DQN training for 15,000
episodes with a schedule that trained the GRU and Q head jointly for 10,000
episodes, then froze the recurrent weights and trained only the Q head for the
remaining 5,000. Nine paired seeds (18–26) ran serially with arm order
alternated by seed. The DQN settings matched the main experiment: 600 steps per
episode, hidden size 64, batch 32, replay capacity 100,000, AdamW learning rate
1e-4, optimization every four environment steps, target copies every 50 steps,
epsilon floor 0.1 at episode 3,750, and automaton-transition action
probability 0.2.

| Measure | Full GRU | Frozen after 10k | Paired result |
|:---|---:|---:|---:|
| Mean process time | 2,331.979 s | 2,122.160 s | 209.820 s saved (95% CI 205.190–214.449) |
| Mean time saved | — | — | 9.00% (95% CI 8.80%–9.19%) |
| Mean final return | 0.507 | 0.501 | Frozen − full: −0.006 (95% CI −0.136 to +0.124) |

Time savings passed its positive-saving criterion. The policy-quality interval
did not meet the predeclared non-inferiority condition (lower bound greater
than −0.05), so this ablation shows a repeatable time reduction but does not
establish preserved policy quality. All nine seeds passed the checks for
matching experience through episode 10,000, bitwise-frozen online and target
GRU tensors, and changed Q-head tensors after freezing.

The 36-checkpoint evaluation used the same 1,900 rollout scenarios per policy
(100 from each traversable start, 1,000-step horizon) and took 5,871.590 s
(1 h 37 min 52 s). Full seed-level timings and paired returns are in
[GRU_FREEZE_EXPERIMENT.md](GRU_FREEZE_EXPERIMENT.md); machine-readable artifacts
are in `artifacts/gru_freeze_ab/confirmatory_10k/`. The paired figure is
![freeze comparison](artifacts/gru_freeze_ab/confirmatory_10k/comparison.png),
and the [policy-map index](artifacts/gru_freeze_ab/confirmatory_10k/policies/index.md)
links the action/value maps for all 36 checkpoints.

### Direct stop at 12,500 episodes

A single additional serial pair (seed 27) measured a direct stop after 10,000
joint GRU/Q-head episodes plus 2,500 head-only episodes. The full 15,000-episode
arm took 2,440.043 s; the frozen 12,500-episode arm took 1,845.933 s, saving
594.110 s (24.35%). Both schedules reached the epsilon floor at episode 3,750.
On 1,900 shared evaluation rollouts, mean return was 0.557321 for the frozen
12.5k policy and 0.460117 for full 15k, a paired difference of +0.097204
(rollout-scenario 95% CI [+0.077363, +0.117045]). This is one training seed;
the interval does not capture seed variation. The result demonstrates an
early-stop time saving and a competitive checkpoint for this run, while the
nine-seed quality-preservation result above remains inconclusive. The full
protocol, timestamps, hyperparameters, and scope notes are in
[GRU_FREEZE_EXPERIMENT.md](GRU_FREEZE_EXPERIMENT.md); see the [analysis](artifacts/gru_freeze_ab/direct_stop_12500_matched/direct_stop_analysis.json)
and [policy maps](artifacts/gru_freeze_ab/direct_stop_12500_matched/policies/index.md).

### One-seed convergence extension

One serial seed-27 pair extended both arms to 25,000 episodes. Stopping the
frozen-GRU run at 20,000 took 3,103.937 s versus 3,970.529 s for the full-GRU
25k run, saving 866.593 s (21.83%). Its return was 0.4542 versus 0.4142 for
full-GRU 25k; paired difference +0.0400 (95% rollout-scenario CI
[+0.0252, +0.0548]). At the same 25k budget, freezing saved 283.184 s
(7.13%). However, the frozen policy fell from 0.4542 at 20k to 0.3275 at
25k, so it failed the prespecified ±0.02 plateau criterion. This demonstrates
an early competitive stopping point and wall-time reduction for one training
seed, but it does not show stable head-only convergence. The nine-seed result
still does not establish policy-quality preservation across seeds. See the
[25k paired analysis](artifacts/gru_freeze_ab/convergence_25k/analysis.json),
[trajectory plot](artifacts/gru_freeze_ab/convergence_25k/convergence.png),
and [policy maps](artifacts/gru_freeze_ab/convergence_25k/policies/index.md).

## Does the GRU receive the automaton state? q-input ablation

The original no-history MLP did not receive q: like the GRU, it saw the task
pair and noisy position. The GRU's additional recurrent inputs are previous
actions and its own remembered epsilon target; these are not the true current
automaton state. A controlled seed-17 run therefore gave q to an MLP while
keeping the same noisy position channel and the same global action interface.
This avoids changing the input and the true-state legal-action mask together.

The MLP-DQN was trained for 15,000 episodes × 600 steps (9 million
environment steps), serially for 1,820.729 s (30 min 20.729 s), with AdamW
learning rate 1e-4, batch 32, replay capacity 100,000, replay optimization
every four steps, target copies every 50 steps, and epsilon reduced from 1.0 to
0.1 over 25% of episodes. Position observation error was 0.25; q was read
exactly; unavailable epsilon actions remained no-ops under the shared global
action set. The two 64-unit-hidden-layer MLPs had 5,640 parameters with q and
5,384 without q.

| Policy | Mean return | 95% rollout-scenario CI |
|:--|--:|:--|
| History-4 GRU-DQN, seed 17 | 0.57160 | [0.54944, 0.59375] |
| Current-observation MLP, q hidden | 0.10526 | [0.09146, 0.11906] |
| MLP with true q and noisy position | 0.10368 | [0.08997, 0.11739] |
| Exact value iteration, inferred q and noisy position | 0.39890 | [0.37688, 0.42092] |
| Exact value iteration, true q and noisy position | 0.53306 | [0.51063, 0.55550] |
| Full-state VI oracle | 0.76462 | [0.74562, 0.78362] |

Every policy used the same 1,900 evaluation scenarios (100 per traversable
start, 1,000-step horizon). Adding q to the MLP changed return by −0.00158
(paired CI [−0.02097, +0.01780]), so it did not help in this seed. GRU beat
the true-q VI policy by +0.03853 (paired CI [+0.01546, +0.06160]) and the
q-observed MLP by +0.46792 ([+0.44548, +0.49036]). The exact q-aware VI result
shows the MLP's low score is a training outcome, not a ceiling imposed by its
feedforward inputs. These intervals represent rollout variation, not
training-seed uncertainty; the experiment uses one training seed.

An initial q-aware MLP run used the true product state's legal-action mask as
well as q and scored 0.21210. Since it changed both the observation and the
action interface, it is only an exploratory diagnostic; its data remain in
`artifacts/pomdp_automaton_state_ablation/evaluation.json`. The controlled
comparison, full 5×4 per-start return tables, exact timings, and analysis are
in [POMDP_AUTOMATON_STATE_ABLATION.md](POMDP_AUTOMATON_STATE_ABLATION.md).
The [notebook-style return grids and action/Q maps](artifacts/pomdp_automaton_state_ablation/index.md)
show where the policies succeed on the task layout. Raw matched rollout data
are in [evaluation_q_observed.json](artifacts/pomdp_automaton_state_ablation/evaluation_q_observed.json).
