"""Run the Safe Absorbing States partial-observation GRU-DQN experiment."""

from __future__ import annotations

import argparse
import random

import numpy as np
import torch

from controlsyn import ControlSynthesis
from mdp import GridMDP
from oa import OmegaAutomaton


STRUCTURE = np.array([
    ["E", "E", "E", "E"],
    ["E", "E", "E", "T"],
    ["B", "E", "E", "E"],
    ["T", "E", "T", "E"],
    ["E", "E", "E", "E"],
])
LABEL = np.array([
    [(),       (),     ("c",), ()],
    [(),       (),     ("a",), ("b",)],
    [(),       (),     ("c",), ()],
    [("b",),  (),     ("a",), ()],
    [(),       ("c",), (),     ("c",)],
], dtype=object)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=800_000)
    parser.add_argument("--horizon", type=int, default=100)
    parser.add_argument("--history-length", type=int, default=16)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--checkpoint-dir", default="checkpoints_pomdp")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    grid_mdp = GridMDP(
        shape=(5, 4),
        structure=STRUCTURE,
        label=LABEL,
        observation_error=0.25,
    )
    oa = OmegaAutomaton("(F G a | F G b) & G !c", "ldba")
    csrl = ControlSynthesis(
        grid_mdp,
        oa,
        discount_nonaccepting=0.99999,
        discount_accepting=0.99,
    )

    print(
        "Starting POMDP GRU-DQN run:",
        f"episodes={args.episodes}, horizon={args.horizon},",
        f"history={args.history_length}, seed={args.seed}",
        flush=True,
    )
    _q_values, _transition_counts, _policy_net = csrl.deep_q_learning(
        T=args.horizon,
        K=args.episodes,
        epsilon_floor=0.1,
        epsilon_decay_fraction=0.25,
        seed=args.seed,
        learning_rate=1e-4,
        automaton_action_probability=0.2,
        architecture="gru",
        history_length=args.history_length,
        gru_hidden_size=64,
        partial_observation=True,
        batch_size=128,
        replay_capacity=100_000,
        target_update_every=1_000,
        checkpoint_dir=args.checkpoint_dir,
        return_diagnostics=True,
        return_model=True,
    )
    print("Training summary:", csrl.last_training_return_summary, flush=True)
    print("Saved checkpoint:", csrl.last_checkpoint_path, flush=True)


if __name__ == "__main__":
    main()
