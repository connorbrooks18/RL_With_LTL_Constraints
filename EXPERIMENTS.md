# Section One experiment protocol

The active Section One study asks whether a history-4 GRU-DQN helps under
partial grid observability and whether GRU freezing or an RSSM encoder changes
training time and convergence. Full results and the same-grid return tables
are in [RESEARCH_RESULTS_REPORT.md](RESEARCH_RESULTS_REPORT.md).

## Observation contract

All learned policies receive the current Rabin-pair index, the true task
automaton state `q`, and a noisy grid row and column. Position-observation
error is 0.25. The GRU and RSSM use four recent observations plus preceding
actions; the MLP receives only the current observation. All arms share the
same q-derived action mask. Only grid position is partially observed.

## Training settings

- One agent, `(F G a | F G b) & G !c`, 5×4 grid.
- Five matched seeds: 17–21; 15,000 episodes × 600 steps (9,000,000 steps/run).
- Batch 32; learning rate `1e-4`; replay capacity 100,000.
- Replay updates every 4 environment steps; target weights copied every 50.
- Epsilon decreases linearly from 1.0 to 0.1 over the first 25% of episodes.
- Automaton-action probability 0.2; discounts 0.99999/0.99.
- Runs were executed serially.
- Freeze arm: GRU and Q head trained jointly through 10,000 episodes; only Q
  head trained afterward.
- RSSM arm: deterministic size 16, two 4-class categorical variables, and
  auxiliary observation/reward/KL weights 0.1/0.1/0.01.

The scale loosely follows Li (2024), Chapter 4, §4.5.1, Task 1. It is adapted
to this project's task, grid, noise channel, and architectures, not reproduced.

## Results at a glance

| Comparison | Result |
|:--|:--|
| GRU vs. current-observation MLP | 0.525 vs. 0.324; paired +0.201, 95% interval [+0.076, +0.326]; GRU won 5/5 seeds |
| GRU vs. noisy-input product-MDP VI | VI 0.533; GRU−VI −0.008, 95% interval [−0.134, +0.118]; no established GRU advantage |
| Frozen vs. full GRU time | Mean observed saving 1,503 s, 95% interval [+225, +2,781] |
| Frozen vs. full GRU final quality | −0.052, 95% interval [−0.192, +0.088]; did not pass −0.05 non-inferiority margin |
| RSSM vs. full GRU time | Mean saving −326 s, 95% interval [−1,646, +994]; no reliable saving |
| RSSM vs. full GRU final quality | −0.022, 95% interval [−0.169, +0.124]; non-inferiority not established |

The full-GRU cohort ran before the frozen/RSSM cohort, and measured throughput
changed sharply during the campaign. The time differences are descriptive; a
counterbalanced timing replication is needed for a strong causal speed claim.

### GRU checkpoint returns versus value iteration

![GRU training checkpoints compared with fixed value-iteration performance](figures/gru_vs_value_iteration_returns.png)

This learning-curve comparison evaluates the saved GRU policies at 10k,
12.5k, and 15k episodes. The dashed value-iteration series is one fixed
q-aware policy, not a model being trained. Both are evaluated on the same 20
rollouts per traversable start with a 500-step horizon; the value-iteration
mean on this bank is 0.4984. Regenerate the figure with
`venv/bin/python plot_gru_training_vs_value_iteration.py`.

## Reproduction

Invoke each command serially. The default checkpoint path includes architecture
and seed. Repeat the commands for seeds 17 through 21.

```sh
venv/bin/python train_pomdp.py --architecture gru \
  --reveal-automaton-state --seed 17 --episodes 15000 --horizon 600 \
  --history-length 4 --checkpoint-episodes 10000 12500

venv/bin/python train_pomdp.py --architecture mlp \
  --reveal-automaton-state --seed 17 --episodes 15000 --horizon 600

venv/bin/python train_pomdp.py --architecture gru \
  --reveal-automaton-state --seed 17 --episodes 15000 --horizon 600 \
  --history-length 4 --freeze-gru-after-episodes 10000

venv/bin/python train_pomdp.py --architecture rssm \
  --reveal-automaton-state --seed 17 --episodes 15000 --horizon 600 \
  --history-length 4 --rssm-hidden-size 16 \
  --rssm-stochastic-variables 2 --rssm-categorical-classes 4
```

For checkpoint maps, run `plot_pomdp_policy_checkpoint.py` with the checkpoint
path and an output path. Its renderer is also the visual reference used for
the policy-map gallery in the experiment record.

## Historical material removed from the active tree

The old office-transfer implementation, q-hidden training-length and
automaton-state ablation workflows, one-seed freeze/RSSM pilot analysis, and
their stand-alone reports were removed. The source notebooks and core grid,
LTL/LDBA, DQN, and POMDP modules remain.
