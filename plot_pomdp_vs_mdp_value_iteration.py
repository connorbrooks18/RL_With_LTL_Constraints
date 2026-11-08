"""Plot paired GRU-DQN gains over MDP value iteration under noisy observations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report = json.loads(args.report.read_text(encoding="utf-8"))
    runs = sorted(report["runs"], key=lambda item: item["training_seed"])
    aggregate = report["across_training_seeds"]["paired_difference_gru_minus_vi"]
    seeds = [item["training_seed"] for item in runs]
    differences = [item["paired_difference_gru_minus_vi"]["mean"] for item in runs]
    rollout_intervals = [item["paired_difference_gru_minus_vi"]["ci95"] for item in runs]

    figure, ax = plt.subplots(figsize=(9.2, 5.2))
    for y, seed, difference, interval in zip(
        range(len(runs)), seeds, differences, rollout_intervals
    ):
        ax.errorbar(
            difference, y,
            xerr=[[difference - interval[0]], [interval[1] - difference]],
            fmt="o", color="#1769aa", capsize=3, markersize=5,
            linewidth=1.2, label="Seed paired mean ± 95% rollout CI" if y == 0 else None,
        )

    mean = aggregate["mean"]
    lo, hi = aggregate["ci95_across_training_seeds"]
    ax.errorbar(
        mean, len(runs) + 0.35, xerr=[[mean - lo], [hi - mean]],
        fmt="s", color="#d95f02", capsize=5, markersize=7,
        linewidth=2, label="Across-seed mean ± 95% Student-t CI",
    )
    ax.axvline(0.0, color="#444444", linestyle="--", linewidth=1)
    ax.set_yticks(list(range(len(runs) + 1)),
                  [f"seed {seed}" for seed in seeds] + ["mean across seeds"])
    ax.set_xlabel("Discounted return difference: history-4 GRU − noisy-input MDP VI")
    ax.set_title(
        "GRU-DQN vs value iteration indexed by noisy state observations\n"
        f"GRU wins on {sum(value > 0 for value in differences)}/{len(differences)} training seeds"
    )
    ax.grid(axis="x", alpha=.25)
    handles, labels = ax.get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=2,
                  bbox_to_anchor=(.5, -.02), fontsize=8)
    figure.tight_layout(rect=(0, .06, 1, 1))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
