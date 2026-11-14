"""Train q-observed DQN policies in Li's single-agent 4x4 office POMDP."""

from __future__ import annotations

import argparse
import hashlib
import random
from pathlib import Path

import numpy as np
import torch

from controlsyn import ControlSynthesis
from oa import OmegaAutomaton
from office_pomdp import (
    OfficeGridMDP,
    SIGNATURE_GROUPS,
    WALL_SIGNATURES,
    read_state_matrix,
)


TASK_FORMULA = "G F (p & F (a | c)) & G !s"
START = (3, 3)  # end of the hallway in Li's Figure 11 / supplementary model


def build_environment(discount_nonaccepting=0.99999, discount_accepting=0.99):
    mdp = OfficeGridMDP(observation_error=0.0)
    oa = OmegaAutomaton(TASK_FORMULA, "ldba")
    csrl = ControlSynthesis(
        mdp,
        oa,
        discount_nonaccepting=discount_nonaccepting,
        discount_accepting=discount_accepting,
    )
    return mdp, oa, csrl


def office_metadata():
    module_path = Path(__file__).with_name("office_pomdp.py")
    script_path = Path(__file__)
    return {
        "environment_name": "Li 4x4 office abstraction (Figure 11; single agent)",
        "office_geometry_source": (
            "https://github.com/JunchaoLi001/LDGBA-Model_Checking/blob/main/"
            "Office%20Simulation/Office%20full%20obsv%20by%20SARSOP/csrl/pomdp.py"
        ),
        "task_formula": TASK_FORMULA,
        "start_position": list(START),
        "office_observation_model": (
            "read_state obtains the complete N,S,E,W local architectural signature; "
            "a unique signature returns its cell exactly, while a repeated signature "
            "returns uniformly among cells with that signature"
        ),
        "office_signature_categories": {
            "0": "hallway", "1": "wall", "2": "door", "3": "window"
        },
        "office_signature_direction_order": ["north", "south", "east", "west"],
        "office_local_signatures": WALL_SIGNATURES.tolist(),
        "office_state_index_order": "row-major (row * 4 + column)",
        "office_read_state_probability_matrix": read_state_matrix().tolist(),
        "office_environment_module_sha256": hashlib.sha256(module_path.read_bytes()).hexdigest(),
        "office_training_script_sha256": hashlib.sha256(script_path.read_bytes()).hexdigest(),
        "office_signature_alias_groups": [
            {
                "signature": list(signature),
                "cells": [list(cell) for cell in cells],
                "observation_probability_per_cell": 1.0 / len(cells),
            }
            for signature, cells in SIGNATURE_GROUPS.items()
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--architecture", choices=("gru", "mlp"), required=True)
    parser.add_argument("--episodes", type=int, default=60_000)
    parser.add_argument("--horizon", type=int, default=600)
    parser.add_argument("--history-length", type=int, default=4)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--replay-capacity", type=int, default=100_000)
    parser.add_argument("--replay-update-every", type=int, default=4)
    parser.add_argument("--target-update-every", type=int, default=50)
    parser.add_argument("--epsilon-floor", type=float, default=0.1)
    parser.add_argument("--epsilon-decay-fraction", type=float, default=0.25)
    parser.add_argument("--discount-nonaccepting", type=float, default=0.99999)
    parser.add_argument("--discount-accepting", type=float, default=0.99)
    parser.add_argument("--checkpoint-dir")
    parser.add_argument("--checkpoint-episodes", type=int, nargs="*", default=())
    args = parser.parse_args()
    if args.checkpoint_dir is None:
        args.checkpoint_dir = (
            f"checkpoints_office_{args.architecture}_q_observed_"
            f"history{args.history_length}_seed{args.seed}_60k"
        )
    if any(episode >= args.episodes for episode in args.checkpoint_episodes):
        parser.error("all intermediate checkpoint episodes must be below --episodes")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    _mdp, _oa, csrl = build_environment(
        discount_nonaccepting=args.discount_nonaccepting,
        discount_accepting=args.discount_accepting,
    )

    print(
        f"Starting office POMDP {args.architecture.upper()}-DQN run:",
        f"episodes={args.episodes}, horizon={args.horizon},",
        f"history={args.history_length}, q_observed=True, seed={args.seed},",
        f"start={START}, task={TASK_FORMULA}",
        flush=True,
    )
    csrl.deep_q_learning(
        start=START,
        T=args.horizon,
        K=args.episodes,
        epsilon_floor=args.epsilon_floor,
        epsilon_decay_fraction=args.epsilon_decay_fraction,
        seed=args.seed,
        learning_rate=args.learning_rate,
        automaton_action_probability=0.2,
        architecture=args.architecture,
        history_length=args.history_length,
        gru_hidden_size=64,
        partial_observation=True,
        automaton_state_observed=True,
        batch_size=args.batch_size,
        replay_update_every=args.replay_update_every,
        replay_capacity=args.replay_capacity,
        target_update_every=args.target_update_every,
        checkpoint_dir=args.checkpoint_dir,
        return_diagnostics=True,
        return_model=True,
        additional_checkpoint_episodes=args.checkpoint_episodes,
        metadata_extra={
            "automaton_state_observed": True,
            "position_observation_error": 0.0,
            **office_metadata(),
        },
    )
    print("Training summary:", csrl.last_training_return_summary, flush=True)
    print("Saved checkpoint:", csrl.last_checkpoint_path, flush=True)
    for path in csrl.last_progress_checkpoint_paths:
        print("Saved intermediate checkpoint:", path, flush=True)


if __name__ == "__main__":
    main()
