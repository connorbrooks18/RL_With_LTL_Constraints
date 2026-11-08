"""Analyze paired multi-length GRU-DQN evaluation reports."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


T_CRITICAL_95 = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776,
                 6: 2.571, 7: 2.447, 8: 2.365, 9: 2.306, 10: 2.262}


def mean_interval(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    se = float(values.std(ddof=1) / np.sqrt(values.size)) if values.size > 1 else 0.0
    return {"mean": mean, "ci95": [mean - 1.96 * se, mean + 1.96 * se],
            "n_rollouts": int(values.size)}


def across_seed_interval(values: list[float]) -> dict:
    array = np.asarray(values, dtype=float)
    mean = float(array.mean())
    n = int(array.size)
    critical = T_CRITICAL_95.get(n, 1.96) if n > 1 else None
    half = (critical * float(array.std(ddof=1)) / np.sqrt(n)
            if critical is not None else None)
    interval = [mean - half, mean + half] if half is not None else None
    return {"mean": mean, "ci95": interval, "n_training_seeds": n}


def collect_samples(run: dict) -> tuple[list[list[int]], np.ndarray]:
    starts = run["starts"]
    squares = [item["square"] for item in starts]
    samples = []
    for item in starts:
        samples.extend(item["rollout_samples"]["gru_dqn"])
    return squares, np.asarray(samples, dtype=float)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path,
                        help="evaluation report containing multiple lengths per seed")
    parser.add_argument("--output", type=Path, required=True,
                        help="write the paired comparison JSON here")
    parser.add_argument("--plot", type=Path, required=True,
                        help="write return and paired-difference figure here")
    args = parser.parse_args()

    report = json.loads(args.report.read_text(encoding="utf-8"))
    by_seed: dict[int, dict[int, dict]] = {}
    for run in report["runs"]:
        match = re.search(r"_K(\d+)_", run["run_id"])
        if not match:
            raise ValueError(f"Cannot find episode count in run ID {run['run_id']}")
        episodes = int(match.group(1))
        seed = int(run["training_seed"])
        if episodes in by_seed.setdefault(seed, {}):
            raise ValueError(f"Duplicate {episodes}-episode run for seed {seed}")
        by_seed[seed][episodes] = run

    complete_seeds = [by_seed[seed] for seed in sorted(by_seed)]
    if not complete_seeds:
        raise ValueError("No training-seed evaluation reports were found")
    lengths = sorted(set.intersection(*(set(entries) for entries in complete_seeds)))
    if len(lengths) < 2:
        raise ValueError("At least two common training lengths are required")
    seeds = sorted(seed for seed, entries in by_seed.items()
                   if all(length in entries for length in lengths))
    if not seeds:
        raise ValueError("No training seed has reports for every common length")

    records = []
    for seed in seeds:
        runs = {length: by_seed[seed][length] for length in lengths}
        reference = runs[lengths[0]]
        sample_arrays = {}
        reference_squares = None
        for length, run in runs.items():
            if run["evaluation_seed"] != reference["evaluation_seed"]:
                raise ValueError(f"Evaluation seeds differ for training seed {seed}")
            squares, samples = collect_samples(run)
            if reference_squares is None:
                reference_squares = squares
            elif squares != reference_squares:
                raise ValueError(f"Start cells differ for seed {seed}, length {length}")
            if length != lengths[0] and samples.size != sample_arrays[lengths[0]].size:
                raise ValueError(f"Rollout counts differ for seed {seed}, length {length}")
            sample_arrays[length] = samples

        adjacent = {}
        for before, after in zip(lengths, lengths[1:]):
            delta = sample_arrays[after] - sample_arrays[before]
            adjacent[f"{before}_to_{after}"] = {
                "mean_difference": float(delta.mean()),
                "paired_difference_rollout_ci": mean_interval(delta),
            }
        records.append({
            "training_seed": seed,
            "scenario_count": int(sample_arrays[lengths[0]].size),
            "mean_return_by_length": {
                str(length): mean_interval(sample_arrays[length])
                for length in lengths
            },
            "adjacent_length_differences": adjacent,
        })

    across_lengths = {
        str(length): across_seed_interval([
            item["mean_return_by_length"][str(length)]["mean"] for item in records
        ])
        for length in lengths
    }
    across_differences = {}
    for before, after in zip(lengths, lengths[1:]):
        key = f"{before}_to_{after}"
        across_differences[key] = across_seed_interval([
            item["adjacent_length_differences"][key]["mean_difference"]
            for item in records
        ])

    output = {
        "comparison": "Same-seed GRU-DQN policies on paired rollout scenarios across training lengths",
        "lengths_episodes": lengths,
        "inference_note": (
            "Rollout intervals are conditional on each trained checkpoint. "
            "Across-seed intervals are Student-t intervals over independent "
            "training-seed means or paired means."
        ),
        "evaluation_seed": report["runs"][0]["evaluation_seed"],
        "rollouts_per_seed": records[0]["scenario_count"],
        "runs": records,
        "across_training_seeds": {
            "mean_return_by_length": across_lengths,
            "adjacent_length_differences": across_differences,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")

    figure, axes = plt.subplots(1, 2, figsize=(13.5, max(5.2, .45 * len(seeds) * (len(lengths) - 1) + 2.3)))
    length_x = np.arange(len(lengths))
    for record in records:
        values = [record["mean_return_by_length"][str(length)]["mean"]
                  for length in lengths]
        axes[0].plot(length_x, values, marker="o", alpha=.8,
                     label=f"seed {record['training_seed']}")
    axes[0].set_xticks(length_x, [f"{length // 1000}k" for length in lengths])
    axes[0].set_ylabel("Mean discounted return")
    axes[0].set_title("Paired policies by training seed")
    axes[0].grid(axis="y", alpha=.25)
    axes[0].legend(fontsize=8, ncol=2)

    y_positions, y_labels = [], []
    row = 0
    colors = plt.cm.tab10(np.linspace(0, 1, max(1, len(seeds))))
    for pair_index, (before, after) in enumerate(zip(lengths, lengths[1:])):
        key = f"{before}_to_{after}"
        for seed_index, record in enumerate(records):
            item = record["adjacent_length_differences"][key]
            delta = item["mean_difference"]
            lo, hi = item["paired_difference_rollout_ci"]["ci95"]
            axes[1].errorbar(delta, row, xerr=[[delta - lo], [hi - delta]],
                             fmt="o", capsize=3, color=colors[seed_index])
            y_positions.append(row)
            y_labels.append(f"{before // 1000}k→{after // 1000}k · seed {record['training_seed']}")
            row += 1
        summary = across_differences[key]
        mean = summary["mean"]
        lo, hi = summary["ci95"] if summary["ci95"] is not None else (mean, mean)
        axes[1].errorbar(mean, row, xerr=[[mean - lo], [hi - mean]],
                         fmt="s", capsize=5, color="#d95f02", linewidth=2)
        y_positions.append(row)
        y_labels.append(f"{before // 1000}k→{after // 1000}k · mean across seeds")
        row += 1
        if pair_index < len(lengths) - 2:
            row += 1

    axes[1].axvline(0, color="#444", linestyle="--", linewidth=1)
    axes[1].set_yticks(y_positions, y_labels)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Change in discounted return")
    axes[1].set_title("Adjacent-length paired differences\nseed: rollout CI; mean: training-seed CI")
    axes[1].grid(axis="x", alpha=.25)
    figure.suptitle("History-4 GRU-DQN training-length comparison")
    figure.tight_layout()
    args.plot.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.plot, dpi=180, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved analysis {args.output}")
    print(f"Saved plot {args.plot}")
    print(json.dumps(output["across_training_seeds"], indent=2))


if __name__ == "__main__":
    main()
