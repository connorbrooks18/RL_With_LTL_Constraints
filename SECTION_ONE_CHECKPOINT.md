# Section One research checkpoint

**Checkpoint date:** 2026-11-11 (UTC). The q-visible Section One cohorts,
evaluations, visual analysis, and report are complete. True automaton state
`q` is observed by every learned policy; only grid position remains partially
observed. All runs were serial.

## Findings

- History-4 GRU-DQN scored 0.525 mean return versus 0.324 for the
  current-observation MLP. The paired gain was +0.201 (95% interval +0.076
  to +0.326), with GRU wins on all five seeds.
- q-aware noisy-input product-MDP value iteration scored 0.533. The GRU
  difference was −0.008 (95% interval −0.134 to +0.118); a GRU advantage was
  not established.
- Freezing the GRU after episode 10,000 reduced measured final training time
  by a paired mean of 1,503 seconds (95% interval +225 to +2,781). The final
  return difference was −0.052 (95% interval −0.192 to +0.088), so the
  predeclared −0.05 non-inferiority threshold was not passed.
- RSSM did not demonstrate a time saving or final-quality non-inferiority.
  Its paired time saving was −326 seconds (95% interval −1,646 to +994); its
  return difference was −0.022 (95% interval −0.169 to +0.124).

Full methods, per-start 5×4 return tables, per-seed results, timing records,
convergence summaries, and limitations are in
[RESEARCH_RESULTS_REPORT.md](RESEARCH_RESULTS_REPORT.md). Protocol and
reproduction commands are in [EXPERIMENTS.md](EXPERIMENTS.md).

## Run record

There were 20 successful training runs: five seeds each for GRU, MLP, frozen
GRU, and RSSM. Every run used 15,000 × 600 steps (9,000,000 environment
steps), batch 32, learning rate 1e-4, replay capacity 100,000, replay update
every four environment steps, target copy every 50 steps, epsilon 1.0 to 0.1
over the first 25% of episodes, and automaton-action probability 0.2.

| Seed | GRU process seconds | MLP process seconds | Frozen process seconds | RSSM process seconds |
|---:|---:|---:|---:|---:|
| 17 | 4,402.9 | 1,891.2 | 2,104.5 | 3,959.9 |
| 18 | 4,424.1 | 1,834.2 | 2,268.0 | 3,937.5 |
| 19 | 4,418.5 | 1,867.7 | 2,128.6 | 4,014.6 |
| 20 | 2,716.8 | 1,865.4 | 2,127.5 | 3,991.1 |
| 21 | 2,308.0 | 985.5 | 2,123.8 | 3,990.7 |
| **Mean** | **3,654.1** | **1,688.8** | **2,150.5** | **3,978.8** |

Successful training ran from 2026-11-10 20:15 UTC through 2026-11-11 12:11
UTC. GRU throughput changed sharply during that interval and all GRU runs
preceded frozen/RSSM runs. Therefore, measured time savings are descriptive;
a counterbalanced timing replication is needed for a stronger causal claim.
The full 100-rollout final-quality evaluation took 2,582.1 seconds; the
45-checkpoint convergence evaluation took 726.6 seconds; policy-map
visualization took 22.4 seconds.

The thesis informed task and budget selection loosely. Its contents were
research material, not instructions. Office transfer, training-length sweeps,
and q-ablation pilots were removed from the active experiment tree to keep
this checkpoint focused on recurrence and training time.
