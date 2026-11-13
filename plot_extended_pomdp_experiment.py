"""Plot extended q-visible GRU/MLP training and checkpoint evaluations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch


def training_plot(checkpoints, output, window):
    colors = {"gru": "#2474b5", "mlp": "#e68632"}
    fig, ax = plt.subplots(figsize=(10, 5.2))
    for path in checkpoints:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        metadata = checkpoint["metadata"]
        returns = checkpoint["episode_returns"].detach().cpu().numpy().astype(float)
        width = min(window, len(returns))
        moving = np.convolve(returns, np.ones(width) / width, mode="valid")
        x = np.arange(width - 1, len(returns)) + 1
        ax.plot(x, moving, color=colors[metadata["architecture"]], alpha=.62,
                linewidth=1.15,
                label=f"{metadata['architecture'].upper()} seed {metadata['seed']}")
    ax.set_xlabel("Training episode")
    ax.set_ylabel(f"Discounted training return · {window}-episode moving average")
    ax.set_title("q-visible POMDP: training returns by architecture")
    ax.grid(alpha=.22)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def evaluation_plot(report_path, output):
    data = json.loads(report_path.read_text(encoding="utf-8"))
    budgets = sorted(int(value) for value in data["by_training_episodes"])
    labels = [str(value) for value in budgets]
    gru, mlp, differences = [], [], []
    for label in labels:
        summary = data["by_training_episodes"][label]["across_training_seeds"]
        gru.append(summary["mean_gru_dqn"])
        mlp.append(summary["mean_no_gru_dqn"])
        differences.append(summary["paired_difference_gru_minus_no_gru"])
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5))
    x = np.asarray(budgets, dtype=float) / 1000
    for result, label, color in ((gru, "GRU", "#2474b5"), (mlp, "Current-observation MLP", "#e68632")):
        means = np.asarray([item["mean"] for item in result], dtype=float)
        ci = np.asarray([item["ci95_across_training_seeds"] for item in result], dtype=float)
        axes[0].errorbar(x, means, yerr=np.vstack((means - ci[:, 0], ci[:, 1] - means)),
                         marker="o", linewidth=1.7, capsize=3, label=label, color=color)
    diff_means = np.asarray([item["mean"] for item in differences], dtype=float)
    diff_ci = np.asarray([item["ci95_across_training_seeds"] for item in differences], dtype=float)
    axes[1].errorbar(x, diff_means,
                     yerr=np.vstack((diff_means - diff_ci[:, 0], diff_ci[:, 1] - diff_means)),
                     marker="o", linewidth=1.7, capsize=3, color="#426b46")
    axes[1].axhline(0, color="#555555", linestyle="--", linewidth=1)
    axes[1].set_title("Paired GRU − MLP")
    axes[1].set_ylabel("Return difference")
    for ax in axes:
        ax.set_xlabel("Training episodes (thousands)")
        ax.set_xticks(x, [f"{value:g}" for value in x])
        ax.grid(alpha=.22)
    axes[0].set_title("Greedy policy evaluation")
    axes[0].set_ylabel("Mean discounted return")
    axes[0].legend()
    fig.suptitle("q-visible POMDP checkpoint evaluation · mean ± 95% CI across training seeds")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def final_evaluation_plot(report_path, output):
    data = json.loads(report_path.read_text(encoding="utf-8"))
    runs = sorted(data["runs"], key=lambda item: int(item["training_seed"]))
    metric = "paired_difference_gru_minus_no_gru"
    seed_x = np.arange(len(runs), dtype=float)
    seed_means = np.asarray([run[metric]["mean"] for run in runs], dtype=float)
    seed_ci = np.asarray([run[metric]["ci95"] for run in runs], dtype=float)
    aggregate = data["across_training_seeds"][metric]
    aggregate_mean = float(aggregate["mean"])
    aggregate_ci = np.asarray(aggregate["ci95_across_training_seeds"], dtype=float)

    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    ax.errorbar(
        seed_x, seed_means,
        yerr=np.vstack((seed_means - seed_ci[:, 0], seed_ci[:, 1] - seed_means)),
        fmt="o", color="#2474b5", capsize=4, linewidth=1.5,
        label="Per-seed paired 95% rollout CI",
    )
    aggregate_x = len(runs) + .25
    ax.errorbar(
        [aggregate_x], [aggregate_mean],
        yerr=np.asarray([[aggregate_mean - aggregate_ci[0]],
                         [aggregate_ci[1] - aggregate_mean]]),
        fmt="D", color="#a33f2b", capsize=5, linewidth=2.0,
        label="Mean ± 95% CI across training seeds",
    )
    ax.axhline(0, color="#555555", linestyle="--", linewidth=1)
    ax.set_xticks(
        [*seed_x, aggregate_x],
        [f"Seed {run['training_seed']}" for run in runs] + ["Across seeds"],
    )
    ax.set_ylabel("Paired GRU − MLP discounted return")
    ax.set_title("60k episode final evaluation · 100 rollouts per start")
    ax.grid(axis="y", alpha=.22)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-checkpoints", type=Path, nargs="+")
    parser.add_argument("--budget-evaluation", type=Path)
    parser.add_argument("--final-evaluation", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--window", type=int, default=1000)
    args = parser.parse_args()
    if args.training_checkpoints:
        training_plot(args.training_checkpoints, args.output_dir / "training_returns.png", args.window)
    if args.budget_evaluation:
        evaluation_plot(args.budget_evaluation, args.output_dir / "evaluation_by_budget.png")
    if args.final_evaluation:
        final_evaluation_plot(args.final_evaluation, args.output_dir / "final_evaluation_by_seed.png")


if __name__ == "__main__":
    main()
