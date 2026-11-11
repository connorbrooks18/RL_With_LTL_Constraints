"""Plot GRU checkpoint return against a fixed noisy-input VI reference."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from evaluate_pomdp import build_environment
from evaluate_pomdp_vs_value_iteration import _rollout


ROOT = Path(__file__).resolve().parent
CONVERGENCE_JSON = ROOT / "artifacts/q_observed_section_one/convergence/evaluation.json"
OUTPUT_DIR = ROOT / "figures"
OUTPUT_PNG = OUTPUT_DIR / "gru_vs_value_iteration_returns.png"
OUTPUT_JSON = OUTPUT_DIR / "gru_vs_value_iteration_returns.json"


def main() -> None:
    report = json.loads(CONVERGENCE_JSON.read_text(encoding="utf-8"))
    config = report["config"]
    if not config["automaton_state_observed"] or not config["grid_position_partially_observed"]:
        raise ValueError("Expected q-visible, partially observed grid evaluations")

    episodes = (10_000, 12_500, 15_000)
    means, lows, highs = [], [], []
    for episode_count in episodes:
        stats = report["aggregate"][f"gru_K{episode_count}"]["return"]
        means.append(float(stats["mean"]))
        lows.append(float(stats["ci95_across_training_seeds"][0]))
        highs.append(float(stats["ci95_across_training_seeds"][1]))

    rollout_count = int(config["rollouts_per_start"])
    horizon = int(config["horizon"])
    eval_seed = int(config["evaluation_seed"])
    grid, _oa, csrl, starts, *_ = build_environment(0.25)
    vi_values = csrl.value_iteration(threshold=1e-8)
    vi_policy = csrl.greedy_policy(vi_values)
    vi_start_means = []
    for start_index, start in enumerate(starts):
        samples = []
        for rollout_index in range(rollout_count):
            rollout_seed = eval_seed + start_index * 1_000_003 + rollout_index
            samples.append(_rollout(
                csrl, start, rollout_seed, horizon,
                value_iteration_value=vi_values,
                value_iteration_policy=vi_policy,
                automaton_state_observed=True,
            ))
        vi_start_means.append(float(np.mean(samples)))
    vi_mean = float(np.mean(vi_start_means))

    result = {
        "automaton_state_observed": True,
        "grid_position_partially_observed": True,
        "evaluation_seed": eval_seed,
        "rollouts_per_start": rollout_count,
        "horizon": horizon,
        "gru": {
            str(episode_count): {
                "mean_return": mean,
                "ci95_across_training_seeds": [low, high],
            }
            for episode_count, mean, low, high in zip(episodes, means, lows, highs)
        },
        "value_iteration": {
            "mean_return": vi_mean,
            "start_means": vi_start_means,
            "line_semantics": "one fixed q-aware MDP policy evaluated on the same rollout bank",
        },
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    x = np.asarray(episodes, dtype=float) / 1000.0
    yerr = np.asarray([
        np.asarray(means) - np.asarray(lows),
        np.asarray(highs) - np.asarray(means),
    ])
    fig, ax = plt.subplots(figsize=(8.8, 5.6), constrained_layout=True)
    ax.errorbar(
        x, means, yerr=yerr, color="#1769aa", marker="o", linewidth=2.2,
        markersize=7, capsize=4,
        label="History-4 GRU-DQN (95% CI across 5 training seeds)",
    )
    ax.axhline(
        vi_mean, color="#d1495b", linestyle="--", linewidth=2,
        label="q-aware noisy-input MDP value iteration (fixed policy)",
    )
    ax.set_xticks(x, ["10k", "12.5k", "15k"])
    ax.set_xlabel("GRU training episodes")
    ax.set_ylabel("Mean discounted return")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("GRU learning checkpoints vs. value-iteration performance")
    ax.text(
        0.01, 0.015,
        f"Same evaluation bank: {rollout_count} rollouts/start, {horizon}-step horizon; q observed, grid position noisy",
        transform=ax.transAxes, fontsize=8.5, color="#444444",
    )
    ax.grid(alpha=0.25)
    ax.legend(loc="best", frameon=True)
    fig.savefig(OUTPUT_PNG, dpi=180)
    plt.close(fig)
    print(f"VI return: {vi_mean:.6f}; saved {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
