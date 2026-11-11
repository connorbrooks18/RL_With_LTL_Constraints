"""Render final q-observed per-start return grids with the notebook visualizer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from evaluate_pomdp import build_environment


ROOT = Path(__file__).resolve().parent
ARM_LABELS = {
    "gru": "History-4 GRU",
    "mlp": "Current-observation MLP",
    "frozen": "Frozen GRU",
    "rssm": "RSSM encoder",
    "value_iteration": "Noisy-input MDP value iteration",
    "gru_minus_mlp": "GRU minus MLP",
    "gru_minus_vi": "GRU minus value iteration",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--comparison",
        type=Path,
        default=Path("artifacts/q_observed_section_one/history4_gru_vs_mlp.json"),
    )
    parser.add_argument(
        "--final-quality",
        type=Path,
        default=Path("artifacts/q_observed_section_one/final_quality/evaluation.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/q_observed_section_one/final_quality"),
    )
    args = parser.parse_args()
    comparison_path = args.comparison if args.comparison.is_absolute() else ROOT / args.comparison
    quality_path = args.final_quality if args.final_quality.is_absolute() else ROOT / args.final_quality
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    quality = json.loads(quality_path.read_text(encoding="utf-8"))
    grid, *_ = build_environment(0.25)
    grids = {
        key: np.zeros(grid.shape, dtype=float)
        for key in ("gru", "mlp", "frozen", "rssm", "value_iteration")
    }

    for record in comparison["runs"]:
        for start in record["starts"]:
            row, col = start["square"]
            grids["gru"][row, col] += (
                float(start["gru_dqn"]["mean"]) / len(comparison["runs"])
            )
            grids["mlp"][row, col] += (
                float(start["no_gru_dqn"]["mean"]) / len(comparison["runs"])
            )

    final_records = quality["checkpoints"]
    for arm in ("frozen", "rssm"):
        records = [item for item in final_records if item["arm"] == arm]
        for record in records:
            for start in record["start_results"]:
                row, col = start["square"]
                grids[arm][row, col] += float(start["mean"]) / len(records)

    vi_record = next(item for item in final_records if item["arm"] == "gru")
    for start in vi_record["start_results"]:
        row, col = start["square"]
        grids["value_iteration"][row, col] = float(start["vi_mean"])
    grids["gru_minus_mlp"] = grids["gru"] - grids["mlp"]
    grids["gru_minus_vi"] = grids["gru"] - grids["value_iteration"]

    serializable = {
        "automaton_state_observed": True,
        "grid_position_partially_observed": True,
        "evaluation_protocol": {
            "rollouts_per_start": 100,
            "horizon": 1000,
            "training_seeds": quality["config"]["training_seeds"],
        },
        "cell_semantics": "Empirical mean discounted return from each start, not a learned scalar value function.",
        "structure": grid.structure.tolist(),
        "labels": [[list(labels) for labels in row] for row in grid.label],
        "grids": {
            key: [[None if not np.isfinite(value) else float(value) for value in row]
                  for row in values]
            for key, values in grids.items()
        },
    }
    (output_dir / "grid_values.json").write_text(
        json.dumps(serializable, indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        "# Final q-observed per-start return grids", "",
        "Values are empirical mean discounted returns from each start, not a learned scalar state-value function.",
        "Rows and columns follow the task map; `B` is a wall, `T` is a trap, and labels are task propositions.",
        "All maps use the final 100-rollout-per-start, 1,000-step evaluation.", "",
    ]
    for key, values in grids.items():
        image_path = output_dir / f"grid_{key}.png"
        fig, ax = plt.subplots(figsize=(7.2, 7.0), constrained_layout=True)
        is_difference = key.startswith("gru_minus_")
        if is_difference:
            bound = max(0.01, float(np.nanmax(np.abs(values))))
            cmap = "coolwarm"
            vmin, vmax = -bound, bound
            color_label = "Mean return difference"
        else:
            cmap = grid.cmap.copy()
            vmin, vmax = 0.0, max(0.01, float(np.nanmax(values)))
            color_label = "Mean discounted return"
        shown = np.ma.masked_where(grid.structure == "B", values)
        cmap_obj = plt.get_cmap(cmap) if isinstance(cmap, str) else cmap
        cmap_obj = cmap_obj.copy()
        cmap_obj.set_bad("#777777")
        image = ax.imshow(shown, interpolation="nearest", cmap=cmap_obj,
                          vmin=vmin, vmax=vmax)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label=color_label)
        ax.set_xticks(np.arange(grid.shape[1]))
        ax.set_yticks(np.arange(grid.shape[0]))
        ax.set_xticks(np.arange(-.5, grid.shape[1], 1), minor=True)
        ax.set_yticks(np.arange(-.5, grid.shape[0], 1), minor=True)
        ax.xaxis.tick_top()
        ax.grid(which="minor", color="lightgray", linewidth=0.8, alpha=0.7)
        ax.tick_params(which="minor", bottom=False, left=False)
        ax.set_xlim(-.5, grid.shape[1] - .5)
        ax.set_ylim(grid.shape[0] - .5, -.5)
        for row_index, col_index in grid.states():
            if grid.structure[row_index, col_index] == "B":
                ax.text(col_index, row_index, "B", ha="center", va="center",
                        color="white", fontweight="bold")
                continue
            if grid.structure[row_index, col_index] == "T":
                ax.add_patch(plt.Circle((col_index, row_index), 0.45,
                                        fill=False, color="black", lw=1.2))
            labels = list(grid.label[row_index, col_index])
            if labels:
                ax.text(col_index, row_index - .19, ",".join(labels),
                        ha="center", va="center", fontsize=9, fontweight="bold")
            ax.text(col_index, row_index + .20, f"{values[row_index, col_index]:.2f}",
                    ha="center", va="center", fontsize=9)
        ax.set_title(ARM_LABELS[key])
        fig.savefig(image_path, dpi=170)
        plt.close(fig)
        lines.extend([f"## {ARM_LABELS[key]}", "", f"![{ARM_LABELS[key]}]({image_path.name})", ""])
        lines.extend(["| Row / column | 0 | 1 | 2 | 3 |", "|:--|--:|--:|:--|--:|"])
        for row_index, row in enumerate(values):
            cells = []
            for col_index, raw in enumerate(row):
                if grid.structure[row_index, col_index] == "B":
                    cells.append("B")
                    continue
                value = f"{raw:.2f}"
                labels = list(grid.label[row_index, col_index])
                if grid.structure[row_index, col_index] == "T":
                    labels.append("T")
                cells.append(f"{'/'.join(labels)} · {value}" if labels else value)
            lines.append(f"| {row_index} | " + " | ".join(cells) + " |")
        lines.append("")
    (output_dir / "grid_values.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved final per-start grids in {output_dir}")


if __name__ == "__main__":
    main()
