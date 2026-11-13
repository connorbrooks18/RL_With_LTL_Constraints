"""Evaluate matched GRU/MLP training checkpoints at multiple episode budgets."""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from evaluate_pomdp_gru_vs_no_gru import _aggregate, _metadata, evaluate_pair


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gru", type=Path, nargs="+", required=True)
    parser.add_argument("--mlp", type=Path, nargs="+", required=True)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--horizon", type=int, default=500)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    def index(paths):
        indexed = {}
        for path in paths:
            metadata = _metadata(path)
            key = (int(metadata["seed"]), int(metadata["episodes"]))
            if key in indexed:
                parser.error(f"duplicate checkpoint for training seed/budget {key}")
            indexed[key] = path
        return indexed

    gru, mlp = index(args.gru), index(args.mlp)
    if set(gru) != set(mlp):
        parser.error(f"GRU and MLP checkpoint cohorts differ: {set(gru)} vs {set(mlp)}")

    started = datetime.now(timezone.utc)
    start_perf = time.perf_counter()
    grouped = defaultdict(list)
    for seed, episodes in sorted(gru):
        print(f"Evaluating seed={seed}, episodes={episodes} ...", flush=True)
        report = evaluate_pair(
            gru[(seed, episodes)], mlp[(seed, episodes)],
            runs=args.runs, horizon=args.horizon, seed=args.seed,
        )
        grouped[episodes].append(report)
        difference = report["paired_difference_gru_minus_no_gru"]
        print(f"  GRU − MLP={difference['mean']:.5f}; rollout CI={difference['ci95']}", flush=True)

    by_episode = {}
    for episodes, reports in sorted(grouped.items()):
        aggregate = _aggregate(reports)
        by_episode[str(episodes)] = {
            "n_training_seeds": len(reports),
            "across_training_seeds": aggregate,
            "runs": reports,
        }
    output = {
        "comparison": "q-visible history-4 GRU-DQN versus current-observation MLP-DQN",
        "episodes_are_from_independent_runs": True,
        "evaluation_design": "paired rollouts from all starts; 20 rollouts per start, 500-step horizon",
        "rollouts_per_start": int(args.runs),
        "horizon": int(args.horizon),
        "evaluation_seed": int(args.seed),
        "started_at_utc": started.isoformat(),
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_elapsed_seconds": time.perf_counter() - start_perf,
        "by_training_episodes": by_episode,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
