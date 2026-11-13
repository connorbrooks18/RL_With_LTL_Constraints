"""Render q-observed GRU/MLP paired evaluation on the original task grid."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from evaluate_pomdp import build_environment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "evaluation", type=Path,
        help="JSON output from evaluate_pomdp_gru_vs_no_gru.py",
    )
    parser.add_argument("--output-dir", type=Path,
                        default=Path("artifacts/q_observed_section_one/history4_comparison"))
    args = parser.parse_args()
    data = json.loads(args.evaluation.read_text(encoding="utf-8"))
    if not data.get("automaton_state_observed"):
        raise SystemExit("Expected a q-observed GRU-versus-MLP evaluation")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    grid, *_ = build_environment(0.25)

    metrics = {
        "gru": ("gru_dqn", "History-4 GRU mean return"),
        "mlp": ("no_gru_dqn", "Current-observation MLP mean return"),
        "gru_minus_mlp": (
            "paired_difference_gru_minus_no_gru",
            "Paired GRU minus MLP return",
        ),
    }
    matrices = {}
    per_seed_matrices = {}
    for key, (field, title) in metrics.items():
        matrix = np.full(grid.shape, np.nan, dtype=float)
        values_by_square = {}
        for run in data["runs"]:
            for start in run["starts"]:
                square = tuple(map(int, start["square"]))
                values_by_square.setdefault(square, []).append(float(start[field]["mean"]))
        for square, values in values_by_square.items():
            matrix[square] = float(np.mean(values))
        matrices[key] = matrix
        visual_values = np.nan_to_num(matrix, nan=0.0)
        # Small paired differences otherwise render in scientific notation
        # and crowd the task-grid cell labels. Keep full precision in JSON.
        if key == "gru_minus_mlp":
            visual_values = np.round(visual_values, 2)
            visual_values[np.abs(visual_values) < 0.005] = 0.0
        grid.plot(value=visual_values, save=str(args.output_dir / f"{key}.png"))
        plt.close(plt.gcf())

    for run in data["runs"]:
        training_seed = int(run["training_seed"])
        per_seed_matrices[str(training_seed)] = {}
        for key, field, title in (
            ("gru", "gru_dqn", "History-4 GRU"),
            ("mlp", "no_gru_dqn", "Current-observation MLP"),
        ):
            matrix = np.full(grid.shape, np.nan, dtype=float)
            for start in run["starts"]:
                square = tuple(map(int, start["square"]))
                matrix[square] = float(start[field]["mean"])
            per_seed_matrices[str(training_seed)][key] = matrix
            grid.plot(
                value=np.nan_to_num(matrix, nan=0.0),
                save=str(args.output_dir / f"seed{training_seed}_{key}_return.png"),
            )
            plt.close(plt.gcf())

    raw = {
        "automaton_state_observed": True,
        "grid_position_partially_observed": True,
        "training_seeds": [int(run["training_seed"]) for run in data["runs"]],
        "structure": grid.structure.tolist(),
        "labels": [[list(grid.label[r, c]) for c in range(grid.shape[1])]
                   for r in range(grid.shape[0])],
        "state_value_grids": {
            key: [[None if not np.isfinite(value) else float(value) for value in row]
                  for row in matrix]
            for key, matrix in matrices.items()
        },
        "per_training_seed": {
            training_seed: {
                architecture: [
                    [None if not np.isfinite(value) else float(value) for value in row]
                    for row in matrix
                ]
                for architecture, matrix in architectures.items()
            }
            for training_seed, architectures in per_seed_matrices.items()
        },
    }
    (args.output_dir / "state_value_grids.json").write_text(
        json.dumps(raw, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# q-observed GRU versus current-observation MLP", "",
        "Automaton state `q` is visible; grid coordinates are noisy. Each image "
        "uses the original task layout and notebook `GridMDP.plot` renderer. "
        "The values are mean empirical discounted returns from each starting "
        "cell, averaged over matched training seeds.", "",
    ]
    for key, (_field, title) in metrics.items():
        lines.extend([f"## {title}", "", f"![{title}]({key}.png)", ""])
    lines.extend(["## Final policy maps by training seed", ""])
    for training_seed in sorted(per_seed_matrices, key=int):
        lines.extend([
            f"### Seed {training_seed}", "",
            f"![Seed {training_seed} GRU final policy return](seed{training_seed}_gru_return.png)", "",
            f"![Seed {training_seed} MLP final policy return](seed{training_seed}_mlp_return.png)", "",
        ])
    lines.append("Numeric arrays and labels are in [state_value_grids.json](state_value_grids.json).")
    (args.output_dir / "index.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved grid comparison to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
