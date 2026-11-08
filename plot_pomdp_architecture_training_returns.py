"""Plot matched GRU and no-GRU training returns from saved checkpoints."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch


def load_group(paths, expected_architecture):
    curves = []
    seeds = []
    episodes = None
    for path in paths:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        metadata = checkpoint["metadata"]
        if metadata["architecture"] != expected_architecture:
            raise ValueError(f"{path} is {metadata['architecture']}, expected {expected_architecture}")
        values = checkpoint["episode_returns"].cpu().numpy().astype(float)
        if episodes is None:
            episodes = len(values)
        elif len(values) != episodes:
            raise ValueError("Training-return plots require equal episode counts")
        curves.append(values)
        seeds.append(int(metadata["seed"]))
    return np.asarray(curves), seeds


def rolling_mean(curves, window):
    kernel = np.ones(window, dtype=float) / window
    return np.asarray([np.convolve(curve, kernel, mode="valid") for curve in curves])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gru", type=Path, nargs="+", required=True)
    parser.add_argument("--no-gru", type=Path, nargs="+", required=True)
    parser.add_argument("--window", type=int, default=1_000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    gru, gru_seeds = load_group(args.gru, "gru")
    mlp, mlp_seeds = load_group(args.no_gru, "mlp")
    if set(gru_seeds) != set(mlp_seeds):
        parser.error(f"Training seeds differ: GRU={sorted(gru_seeds)}, MLP={sorted(mlp_seeds)}")
    if args.window < 1 or args.window > min(gru.shape[1], mlp.shape[1]):
        parser.error("--window must be in [1, number of episodes]")

    gru_smooth = rolling_mean(gru, args.window)
    mlp_smooth = rolling_mean(mlp, args.window)
    x = np.arange(args.window, gru.shape[1] + 1)
    figure, ax = plt.subplots(figsize=(10.2, 5.8))
    for values, label, color in ((gru_smooth, "History-4 GRU-DQN", "#1769aa"),
                                 (mlp_smooth, "No-history MLP-DQN", "#e58b22")):
        mean = values.mean(axis=0)
        spread = values.std(axis=0, ddof=1) if values.shape[0] > 1 else np.zeros_like(mean)
        ax.plot(x, mean, label=label, color=color, linewidth=2)
        ax.fill_between(x, mean - spread, mean + spread, color=color, alpha=.18,
                        label=f"{label} ±1 SD")
    ax.set_xlabel("Training episode")
    ax.set_ylabel(f"Mean return over previous {args.window} episodes")
    ax.set_title("Matched POMDP training-return curves")
    ax.grid(alpha=.25)
    ax.legend()
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=170, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved {args.output}; training seeds {sorted(gru_seeds)}")


if __name__ == "__main__":
    main()
