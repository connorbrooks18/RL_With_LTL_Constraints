"""Visualize paired GRU versus no-GRU evaluation results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    runs = report["runs"]
    seeds = np.asarray([run["training_seed"] for run in runs])
    differences = np.asarray([
        run["paired_difference_gru_minus_no_gru"]["mean"] for run in runs
    ], dtype=float)
    aggregate = report["across_training_seeds"]["paired_difference_gru_minus_no_gru"]
    lower, upper = aggregate["ci95_across_training_seeds"]

    figure = plt.figure(figsize=(14.5, 6.5))
    layout = figure.add_gridspec(1, 3, width_ratios=(1.2, 1.0, 1.35), wspace=.32)

    ax = figure.add_subplot(layout[0, 0])
    for index, run in enumerate(runs):
        result = run["paired_difference_gru_minus_no_gru"]
        mean = result["mean"]
        lo, hi = result["ci95"]
        ax.errorbar(mean, index, xerr=[[mean - lo], [hi - mean]], fmt="o",
                    color="#1769aa", capsize=3)
    ax.axvline(0, color="#555555", linestyle="--", linewidth=1)
    ax.set_yticks(range(len(runs)), [str(seed) for seed in seeds])
    ax.invert_yaxis()
    ax.set_xlabel("Paired discounted return difference")
    ax.set_ylabel("Training seed")
    ax.set_title("Within-seed paired rollout estimates\n(95% rollout CIs)")
    ax.grid(axis="x", alpha=.22)

    ax = figure.add_subplot(layout[0, 1])
    ax.axvline(0, color="#555555", linestyle="--", linewidth=1)
    ax.errorbar(aggregate["mean"], 0,
                xerr=[[aggregate["mean"] - lower], [upper - aggregate["mean"]]],
                fmt="D", color="#b23a48", capsize=4, markersize=7,
                label="Mean and 95% t CI across seeds")
    jitter = np.linspace(-.16, .16, len(differences)) if len(differences) > 1 else [0]
    ax.scatter(differences, jitter, color="#1769aa", alpha=.8, zorder=3,
               label="Training-seed means")
    ax.set_yticks([])
    ax.set_xlabel("GRU − no-GRU discounted return")
    ax.set_title(f"Across {len(differences)} training seeds\n"
                 f"mean={aggregate['mean']:.3f}; 95% CI [{lower:.3f}, {upper:.3f}]")
    ax.grid(axis="x", alpha=.22)
    ax.legend(fontsize=7, loc="lower right")

    ax = figure.add_subplot(layout[0, 2])
    grid = np.full((5, 4), np.nan, dtype=float)
    labels = np.full((5, 4), "", dtype=object)
    for run in runs:
        for start in run["starts"]:
            row, col = start["square"]
            diff = start["paired_difference_gru_minus_no_gru"]["mean"]
            if labels[row, col] == "":
                labels[row, col] = start["label"]
                grid[row, col] = 0.0
            grid[row, col] += diff / len(runs)
    cmap = plt.cm.RdBu.copy()
    cmap.set_bad("#777777")
    vmax = max(float(np.nanmax(np.abs(grid))), 1e-6)
    image = ax.imshow(grid, cmap=cmap, vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(4))
    ax.set_yticks(range(5))
    ax.set_xticks(np.arange(-.5, 4, 1), minor=True)
    ax.set_yticks(np.arange(-.5, 5, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.3)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.xaxis.tick_top()
    for row in range(5):
        for col in range(4):
            if np.isnan(grid[row, col]):
                ax.text(col, row, "B", ha="center", va="center", color="white", fontsize=9)
            else:
                label = ",".join(labels[row, col])
                ax.text(col, row - .10, f"{grid[row, col]:+.2f}", ha="center", va="center",
                        fontsize=8, fontweight="bold")
                if label:
                    ax.text(col, row + .20, label, ha="center", va="center", fontsize=6)
    ax.set_title("Mean paired difference by start\nsquare on task grid", fontsize=10)
    figure.colorbar(image, ax=ax, fraction=.046, pad=.04, label="GRU − no-GRU")

    figure.suptitle("History-4 GRU-DQN versus feedforward no-history DQN",
                    fontsize=13, y=.99)
    figure.subplots_adjust(left=.055, right=.97, bottom=.12, top=.80, wspace=.36)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=175, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
