"""Plot q0 values and greedy actions from repeated copies of each square's observation.

Example:
    python plot_synthetic_pomdp_history.py --repeats 1 2 4 8 16
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

from dqn import GRUDQN
from evaluate_pomdp import action_name, build_environment


def newest_checkpoint(directory: Path) -> Path:
    checkpoints = list(directory.glob("dqn_*.pt"))
    if not checkpoints:
        raise FileNotFoundError(f"No dqn_*.pt checkpoints found in {directory}")
    return max(checkpoints, key=lambda path: path.stat().st_mtime)


def default_repeats(history_length: int) -> list[int]:
    counts = [1]
    while counts[-1] * 2 < history_length:
        counts.append(counts[-1] * 2)
    if history_length > counts[-1]:
        counts.append(history_length)
    return counts


def plot_history_count(
    count: int,
    checkpoint: Path,
    model: GRUDQN,
    metadata: dict,
    grid,
    oa,
    csrl,
    pair: int,
    output_dir: Path,
) -> Path:
    history_length = int(metadata["history_length"])
    q_grid = np.full(grid.shape, np.nan, dtype=float)
    best_actions: dict[tuple[int, int], int] = {}

    for row, col in grid.states():
        if grid.structure[row, col] == "B":
            continue
        state = (pair, oa.q0, row, col)
        observation = csrl.observe_observation(state, error_prob=0.0)
        observations = [observation] * count
        previous_actions = [None] * count
        encoded = csrl.encode_observation_history(
            observations,
            history_length=history_length,
            previous_actions=previous_actions,
            partial_observation=True,
            epsilon_commitment_mode=metadata.get(
                "recurrent_epsilon_commitment_mode", "none"
            ),
        ).unsqueeze(0)
        with torch.no_grad():
            q_values = model(encoded).squeeze(0).cpu().numpy()

        available = list(csrl.A[state])
        action = max(available, key=lambda candidate: q_values[candidate])
        q_grid[row, col] = float(q_values[action])
        best_actions[(row, col)] = action

    grid.plot(value=q_grid)
    ax = plt.gca()
    arrows = {"U": "↑", "D": "↓", "R": "→", "L": "←"}
    for (row, col), action in best_actions.items():
        name = action_name(action, len(grid.A))
        marker = arrows[name] if name in arrows else rf"$\epsilon_{{{action - len(grid.A)}}}$"
        ax.text(
            col + 0.31,
            row + 0.29,
            marker,
            ha="center",
            va="center",
            fontsize=11,
            fontweight="bold",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 1},
        )

    ax.set_title(
        f"q0: same clean observation repeated {count} time(s)\n"
        f"checkpoint history length={history_length}; number=max Q, symbol=greedy action"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"same_observation_n{count:02d}.png"
    plt.gcf().savefig(output, bbox_inches="tight", dpi=160)
    plt.close(plt.gcf())
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Checkpoint to inspect (defaults to the newest in --checkpoint-dir).",
    )
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints_pomdp"))
    parser.add_argument("--repeats", type=int, nargs="+", help="Observation repeat counts.")
    parser.add_argument("--pair", type=int, default=0, help="LTL pair index (default: 0).")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/synthetic_history")
    )
    args = parser.parse_args()

    checkpoint_path = args.checkpoint or newest_checkpoint(args.checkpoint_dir)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    metadata = checkpoint["metadata"]
    history_length = int(metadata["history_length"])
    repeats = args.repeats or default_repeats(history_length)
    if any(count < 1 or count > history_length for count in repeats):
        parser.error(f"--repeats must be between 1 and checkpoint history length ({history_length})")

    grid, oa, csrl, *_ = build_environment(float(metadata["observation_error"]))
    if not 0 <= args.pair < csrl.shape[0]:
        parser.error(f"--pair must be between 0 and {csrl.shape[0] - 1}")

    model = GRUDQN(
        metadata["n_observations"],
        metadata["n_actions"],
        hidden_size=metadata["gru_hidden_size"],
        num_layers=metadata["gru_num_layers"],
    )
    model.load_state_dict(checkpoint["policy_state_dict"])
    model.eval()

    print(f"Checkpoint: {checkpoint_path}")
    print(f"Repeated-observation counts: {repeats}")
    for count in repeats:
        output = plot_history_count(
            count, checkpoint_path, model, metadata, grid, oa, csrl,
            args.pair, args.output_dir,
        )
        print(f"Saved {output}")


if __name__ == "__main__":
    main()
