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
