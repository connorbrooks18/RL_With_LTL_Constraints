"""Plot episode-return curves for office GRU-DQN checkpoints."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoints", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--window", type=int, default=500)
    args = parser.parse_args()
    figure, ax = plt.subplots(figsize=(9.5, 4.8))
    summaries = []
    for path in args.checkpoints:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        metadata = checkpoint["metadata"]
        returns = checkpoint["episode_returns"].detach().cpu().numpy().astype(float)
        window = min(args.window, len(returns))
        moving = np.convolve(returns, np.ones(window) / window, mode="valid")
        ax.plot(np.arange(window - 1, len(returns)), moving,
                label=f"seed {metadata['seed']}", linewidth=1.25)
        summary = metadata["training_return_summary"]
        summaries.append((metadata["seed"], summary["first_window_mean"],
                          summary["last_window_mean"], summary["mean"]))
    ax.set_xlabel("Training episode")
    ax.set_ylabel("Discounted training return")
    ax.set_title(f"Office GRU-DQN training returns · {args.window}-episode moving average")
    ax.grid(alpha=.22)
    ax.legend(ncol=2, fontsize=8)
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=175, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved {args.output}")
    for seed, first, last, mean in summaries:
        print(f"seed {seed}: mean={mean:.6f}, first1k={first:.6f}, last1k={last:.6f}")


if __name__ == "__main__":
    main()
