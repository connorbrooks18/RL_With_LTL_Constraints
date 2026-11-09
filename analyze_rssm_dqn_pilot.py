"""Analyze matched RSSM-DQN and GRU-DQN pilot evaluations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch


def flatten_samples(report: dict, key: str = "policy_dqn") -> np.ndarray:
    return np.asarray([
        value
        for start in report["starts"]
        for value in start["rollout_samples"][key]
    ], dtype=float)


def label(report: dict) -> str:
    architecture = report["architecture"].upper()
    episodes = int(report["training_episodes"])
    episode_label = f"{episodes / 1000:g}k"
    if architecture == "GRU" and report.get("gru_frozen_at_checkpoint"):
        phase = "frozen"
    else:
        phase = "joint"
    name = "RSSM-inspired encoder" if architecture == "RSSM" else architecture
    return f"{name} {episode_label} ({phase})"


def paired_summary(left: np.ndarray, right: np.ndarray) -> dict:
    difference = left - right
    mean = float(difference.mean())
    stderr = float(difference.std(ddof=1) / np.sqrt(difference.size))
    return {
        "mean_difference": mean,
        "ci95_rollout_scenarios": [mean - 1.96 * stderr, mean + 1.96 * stderr],
        "n_rollout_scenarios": int(difference.size),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evaluation", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(args.evaluation.read_text(encoding="utf-8"))
    reports = source["runs"]
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    entries = []
    training_series = {}
    for report in reports:
        samples = flatten_samples(report)
        vi_samples = flatten_samples(
            report, key="mdp_value_iteration_on_noisy_observation"
        )
        entries.append({
            "label": label(report),
            "architecture": report["architecture"],
            "training_episodes": int(report["training_episodes"]),
            "training_loop_seconds": float(report["training_loop_seconds"]),
            "training_seed": report["training_seed"],
            "checkpoint": report["checkpoint"],
            "mean_return": float(samples.mean()),
            "vi_mean_return": float(vi_samples.mean()),
            "rssm_minus_vi": paired_summary(samples, vi_samples),
            "samples": samples,
        })
        checkpoint = torch.load(report["checkpoint"], map_location="cpu",
                                weights_only=True)
        returns = checkpoint.get("episode_returns")
        if returns is not None:
            series_key = (
                report["architecture"],
                bool(report.get("gru_frozen_at_checkpoint")),
            )
            candidate = returns.detach().cpu().numpy().astype(float)
            previous = training_series.get(series_key)
            if previous is None or candidate.size > previous.size:
                training_series[series_key] = candidate

    pairwise = []
    for i, left in enumerate(entries):
        for right in entries[i + 1:]:
            summary = paired_summary(left["samples"], right["samples"])
            pairwise.append({
                "policy_a": left["label"],
                "policy_b": right["label"],
                "a_minus_b": summary,
            })
    points = [{key: value for key, value in entry.items() if key != "samples"}
              for entry in entries]
    analysis = {
        "interpretation": (
            "Single training seed. Confidence intervals quantify variation over "
            "the shared rollout scenarios, not training-seed uncertainty."
        ),
        "evaluation_seed": source["evaluation_batch"]["evaluation_seed"],
        "rollouts_per_start": source["evaluation_batch"]["rollouts_per_start"],
        "horizon": source["evaluation_batch"]["horizon"],
        "policy_points": points,
        "pairwise_policy_differences": pairwise,
    }
    (output_dir / "analysis.json").write_text(
        json.dumps(analysis, indent=2) + "\n", encoding="utf-8"
    )

    fig, ax = plt.subplots(figsize=(9, 5.5))
    colors = {"rssm": "#7b3294", "gru": "#008837"}
    markers = {"rssm": "o", "gru": "s"}
    for architecture in ("gru", "rssm"):
        group = [entry for entry in entries
                 if entry["architecture"] == architecture]
        if not group:
            continue
        xs = [entry["training_loop_seconds"] / 60 for entry in group]
        ys = [entry["mean_return"] for entry in group]
        ax.plot(xs, ys, color=colors[architecture], marker=markers[architecture],
                linewidth=1.7, label=f"{architecture.upper()}-DQN")
        for entry, x, y in zip(group, xs, ys):
            ax.annotate(entry["label"], (x, y), xytext=(5, 6),
                        textcoords="offset points", fontsize=8)
    ax.set_xlabel("Training loop time (minutes)")
    ax.set_ylabel("Mean return on shared evaluation scenarios")
    ax.set_title("RSSM-DQN training duration versus GRU-DQN checkpoints")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "return_vs_training_time.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for (architecture, frozen), returns in training_series.items():
        window = min(1000, returns.size)
        smoothed = np.convolve(returns, np.ones(window) / window, mode="valid")
        episodes = np.arange(window, returns.size + 1)
        policy_label = architecture.upper()
        if architecture == "gru":
            policy_label += " frozen encoder" if frozen else " full GRU"
        ax.plot(episodes, smoothed, label=policy_label, linewidth=1.4)
    ax.set_xlabel("Training episode")
    ax.set_ylabel("Trailing mean training return")
    ax.set_title("Training return trajectories (1000 episode window where available)")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "training_return_trajectories.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
