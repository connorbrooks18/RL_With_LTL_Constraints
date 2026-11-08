"""Plot office task returns and LTL acceptance metrics on the 4x4 map."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from office_pomdp import LABEL, SHAPE
from train_office_pomdp import START


METRICS = (
    ("discounted_return", "Mean discounted return", "viridis", ".1e"),
    ("completed_ltl_acceptance_cycles", "Mean LTL-acceptance cycles", "viridis", ".2f"),
    ("at_least_two_ltl_cycles_without_storage_violation",
     "P(≥2 LTL cycles, no storage violation)", "magma", ".2f"),
)


def plot(report_path: Path, output_path: Path) -> None:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    entries = report["across_training_seeds"]
    maps = {}
    for metric, _title, _cmap, _fmt in METRICS:
        values = np.full(SHAPE, np.nan, dtype=float)
        for entry in entries:
            row, col = entry["square"]
            values[row, col] = entry[metric]["mean"]
        maps[metric] = values

    figure, axes = plt.subplots(1, len(METRICS), figsize=(15.4, 5.1))
    for ax, (metric, title, cmap, fmt) in zip(axes, METRICS):
        values = maps[metric]
        finite = values[np.isfinite(values)]
        lo, hi = float(finite.min()), float(finite.max())
        if hi <= lo:
            hi = lo + 1e-6
        image = ax.imshow(values, cmap=cmap, vmin=lo, vmax=hi,
                          interpolation="nearest")
        ax.set_xticks(range(SHAPE[1]))
        ax.set_yticks(range(SHAPE[0]))
        ax.set_xticks(np.arange(-.5, SHAPE[1], 1), minor=True)
        ax.set_yticks(np.arange(-.5, SHAPE[0], 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.5)
        ax.tick_params(which="minor", bottom=False, left=False)
        ax.xaxis.tick_top()
        for row in range(SHAPE[0]):
            for col in range(SHAPE[1]):
                value = values[row, col]
                label = tuple(LABEL[row, col])
                proposition = ({"p": "Print", "s": "S"}.get(label[0], label[0])
                               if label else "")
                if np.isfinite(value):
                    ax.text(col, row - .10, format(value, fmt), ha="center",
                            va="center", fontsize=9, fontweight="bold",
                            color="white" if value > lo + .62 * (hi - lo) else "#202020")
                if proposition:
                    ax.text(col, row + .22, proposition, ha="center", va="center",
                            fontsize=7, color="white" if value > lo + .62 * (hi - lo) else "#202020")
                if (row, col) == START:
                    ax.scatter([col], [row], marker="o", s=100,
                               facecolors="none", edgecolors="#00e5d4",
                               linewidths=2, zorder=4)
        ax.set_title(title, fontsize=10)
        figure.colorbar(image, ax=ax, fraction=.047, pad=.04)

    n_seeds = len(report["runs"])
    figure.suptitle(
        f"Single-agent office GRU-DQN · {n_seeds} training seeds · "
        "100 rollouts per start · 1,000-step horizon\n"
        "Values average over training seeds; ring marks thesis start",
        fontsize=11,
    )
    figure.tight_layout(rect=(0, 0, 1, .86))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=175, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plot(args.report, args.output)
    print(f"Saved office evaluation grid {args.output}")


if __name__ == "__main__":
    main()
