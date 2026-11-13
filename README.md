# Temporal-logic motion planning with partial observations

This repository studies deep Q-learning for motion planning under temporal
logic constraints. The current Section One experiments expose the true task
automaton state `q` and keep grid position partially observable.

## Research results

- [Section One research report](RESEARCH_RESULTS_REPORT.md): protocol,
  outcomes, per-start 5×4 returns, wall-clock record, and limitations.
- [Experiment protocol and reproduction commands](EXPERIMENTS.md).
- [Current Section One checkpoint](SECTION_ONE_CHECKPOINT.md).

### GRU training and value-iteration performance

The GRU curve shows greedy policy returns at saved training checkpoints. The
horizontal value-iteration line is a fixed-policy reference evaluated on the
same scenarios; value iteration has no training curve.

![GRU checkpoint returns versus fixed value-iteration performance](figures/gru_vs_value_iteration_returns.png)

### Longer training and office transfer

With true `q` and partial grid observations, the 30k core comparison gives a
GRU-minus-current-observation-MLP mean return gain of **+0.232** (95% interval
+0.190 to +0.273 across three training seeds). This establishes the core
comparison at 30k episodes, though intermediate-budget intervals and training
curves do not establish convergence. The report includes the longer training
curves and same-grid empirical return maps.

The independent 60k core extension has completed on seeds 17–19. At 60k,
mean return was 0.570 for GRU and 0.293 for MLP, with a paired difference of
+0.277; the across-seed 95% interval is wide and includes zero (−0.254 to
+0.807). All three seed means favor GRU, but this extension does not establish
a seed-robust advantage. Across 30k, 45k, and 60k snapshots from this cohort,
only the 45k paired interval excludes zero. The results and training curves
are shown below; the report gives the protocol, timing, and per-seed details.

![60k core evaluation by training budget](figures/section_one_60k/evaluation_by_budget.png)

![60k final evaluation by training seed](figures/section_one_60k/final_evaluation_by_seed.png)

![60k core training returns](figures/section_one_60k/training_returns.png)

The notebook-style policy-grid difference averages GRU-minus-MLP returns over
the 19 evaluated start cells:

![60k core paired return difference on the task grid](figures/section_one_60k/gru_minus_mlp.png)

The 60k office extension also does not establish a recurrent advantage. At
60k, all-room return was 0.00125 for GRU versus 0.00135 for MLP, and safe
acceptance was 0.118 versus 0.126. From thesis start `(3,3)`, safe acceptance
was 0.133 versus 0.200. The three-seed intervals cross zero and seed effects
change direction. Office curves, budget comparisons, and policy maps are in
`figures/office_60k/` and `artifacts/office_extended_60k/visualizations/`.

![Office training returns](figures/office_60k/training_returns.png)

![Office evaluation by training budget](figures/office_60k/evaluation_by_budget.png)

The original full-GRU/frozen-GRU/RSSM training-speed comparison is also
complete. Neither freezing nor RSSM established a reliable wall-clock saving
with preserved final quality. Thus all currently planned experiment runs are
complete; the results do not justify claiming a positive Office transfer or
a proven speedup. Per-policy notebook-style maps for the 60k core extension
are linked in [the results report](RESEARCH_RESULTS_REPORT.md).

## Core implementation

- `controlsyn.py`: product MDP construction and DQN training.
- `dqn.py`: MLP-DQN, GRU-DQN, and RSSM-DQN.
- `train_pomdp.py`: serially invoked POMDP experiment runner. Pass
  `--reveal-automaton-state` to observe `q`; grid observations remain noisy.
- `evaluate_pomdp.py`, `evaluate_pomdp_gru_vs_no_gru.py`, and
  `evaluate_pomdp_vs_value_iteration.py`: policy and baseline evaluation.
- `plot_pomdp_policy_checkpoint.py`: notebook-style policy and action maps.

Run metadata, checkpoints, and generated figures are stored locally under
ignored `artifacts/` and `checkpoints_*/` directories.
