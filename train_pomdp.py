"""Run the Safe Absorbing States partial-observation DQN experiment.

Defaults follow the model-free grid-world Task 1 budget in Chapter 4 of Li
(2024), adapted to this repository's environment.
"""

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
    parser.add_argument("--episodes", type=int, default=15_000)
    parser.add_argument("--horizon", type=int, default=600)
    parser.add_argument("--history-length", type=int, default=4)
    parser.add_argument("--rssm-hidden-size", type=int, default=32)
    parser.add_argument("--rssm-stochastic-variables", type=int, default=4)
    parser.add_argument("--rssm-categorical-classes", type=int, default=4)
    parser.add_argument(
        "--architecture", choices=("gru", "rssm", "mlp"), default="gru",
        help="gru and rssm use an observation history; mlp is feedforward",
    )
    parser.add_argument(
        "--reveal-automaton-state", action="store_true",
        help=("for an MLP, include the true automaton state with the noisy "
              "position while retaining the partial-observation action interface"),
    )
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--replay-update-every", type=int, default=4)
    parser.add_argument("--replay-capacity", type=int, default=100_000)
    parser.add_argument("--target-update-every", type=int, default=50)
    parser.add_argument("--epsilon-floor", type=float, default=0.1)
    parser.add_argument("--epsilon-decay-fraction", type=float, default=0.25)
    parser.add_argument(
        "--freeze-gru-after-episodes", type=int,
        help="freeze the GRU after this many episodes and train only the Q head",
    )
    parser.add_argument("--checkpoint-dir")
    parser.add_argument(
        "--checkpoint-episodes", type=int, nargs="*",
        help="save intermediate policy snapshots at these episode counts",
    )
    args = parser.parse_args()
    if args.reveal_automaton_state and args.architecture != "mlp":
        parser.error("--reveal-automaton-state is only valid with --architecture mlp")
    if args.checkpoint_dir is None:
        if args.architecture in ("gru", "rssm"):
            args.checkpoint_dir = (
                f"checkpoints_pomdp_{args.architecture}_history{args.history_length}_seed{args.seed}"
            )
        elif args.reveal_automaton_state:
            args.checkpoint_dir = f"checkpoints_pomdp_mlp_q_observed_seed{args.seed}"
        else:
            args.checkpoint_dir = f"checkpoints_pomdp_mlp_seed{args.seed}"

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
        f"Starting POMDP {args.architecture.upper()}-DQN run:",
        f"episodes={args.episodes}, horizon={args.horizon},",
        f"architecture={args.architecture}, history={args.history_length}, "
        f"automaton_state_observed={args.reveal_automaton_state}, seed={args.seed}",
        flush=True,
    )
    progress_checkpoints = args.checkpoint_episodes or (
        range(args.freeze_gru_after_episodes + 2_500, args.episodes, 2_500)
        if args.freeze_gru_after_episodes is not None else ()
    )
    _q_values, _transition_counts, _policy_net = csrl.deep_q_learning(
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
        automaton_state_observed=args.reveal_automaton_state,
        batch_size=args.batch_size,
        replay_update_every=args.replay_update_every,
        replay_capacity=args.replay_capacity,
        target_update_every=args.target_update_every,
        checkpoint_dir=args.checkpoint_dir,
        return_diagnostics=True,
        return_model=True,
        freeze_gru_after_episodes=args.freeze_gru_after_episodes,
        additional_checkpoint_episodes=progress_checkpoints,
        metadata_extra={
            "automaton_state_observed": bool(args.reveal_automaton_state),
            "position_observation_error": float(grid_mdp.observation_error),
        },
        rssm_deterministic_size=args.rssm_hidden_size,
        rssm_stochastic_variables=args.rssm_stochastic_variables,
        rssm_categorical_classes=args.rssm_categorical_classes,
    )
    print("Training summary:", csrl.last_training_return_summary, flush=True)
    print("Saved checkpoint:", csrl.last_checkpoint_path, flush=True)
    if args.freeze_gru_after_episodes is not None:
        print(
            "Saved pre-freeze checkpoint:",
            csrl.last_freeze_snapshot_path,
            flush=True,
        )
        for path in csrl.last_progress_checkpoint_paths:
            print("Saved head-only progress checkpoint:", path, flush=True)


if __name__ == "__main__":
    main()
