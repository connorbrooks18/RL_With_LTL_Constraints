"""Analyze one-seed frozen-GRU convergence against a matched full-GRU run."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch


ROOT = Path(__file__).resolve().parent
PLATEAU_TOLERANCE = 0.02
QUALITY_MARGIN = 0.05


def ci95(values):
    x = np.asarray(values, dtype=float)
    mean = float(x.mean())
    se = float(x.std(ddof=1) / np.sqrt(x.size)) if x.size > 1 else 0.0
    return {"mean": mean, "ci95": [mean - 1.96 * se, mean + 1.96 * se],
            "n_rollouts": int(x.size)}


def rollout_samples(run):
    squares = [item["square"] for item in run["starts"]]
    gru = np.asarray([
        value for item in run["starts"]
        for value in item["rollout_samples"]["gru_dqn"]
    ], dtype=float)
    vi = np.asarray([
        value for item in run["starts"]
        for value in item["rollout_samples"][
            "mdp_value_iteration_on_noisy_observation"
        ]
    ], dtype=float)
    return squares, gru, vi


def checkpoint(path):
    return torch.load(ROOT / path, map_location="cpu", weights_only=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("evaluation", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plot", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    evaluation = json.loads(args.evaluation.read_text(encoding="utf-8"))
    completed = [item for item in manifest["runs"]
                 if item.get("status") == "complete"]
    by_arm = {item["arm"]: item for item in completed}
    if set(by_arm) != {"full", "frozen"}:
        raise ValueError("Expected one complete full and frozen run")
    full = by_arm["full"]
    frozen = by_arm["frozen"]
    eval_by_id = {item["run_id"]: item for item in evaluation["runs"]}

    freeze_episode = int(manifest["config"]["freeze_after_episodes"])
    pre_metadata_path = ROOT / frozen["pre_freeze_checkpoint"]
    pre_metadata_path = pre_metadata_path.with_suffix(".json")
    pre_metadata = json.loads(pre_metadata_path.read_text(encoding="utf-8"))
    frozen_points = {int(frozen["episodes"]): {
        "run_id": frozen["run_id"],
        "checkpoint": frozen["checkpoint"],
        "training_loop_seconds": frozen["training_loop_seconds"],
    }, freeze_episode: {
        "run_id": frozen["pre_freeze_run_id"],
        "checkpoint": frozen["pre_freeze_checkpoint"],
        "training_loop_seconds": pre_metadata["training_loop_seconds"],
    }}
    for item in frozen.get("head_only_progress_checkpoints", []):
        frozen_points[int(item["episodes"])] = item
    wanted_episodes = (10_000, 15_000, 20_000, 25_000)
    if any(ep not in frozen_points for ep in wanted_episodes):
        raise ValueError("Frozen policy checkpoints missing one or more planned endpoints")
    if int(full["episodes"]) != 25_000 or int(frozen["episodes"]) != 25_000:
        raise ValueError("This analysis plan requires both arms to run 25,000 episodes")

    run_by_episode = {}
    for episode in wanted_episodes:
        point = frozen_points[episode]
        if point["run_id"] not in eval_by_id:
            raise ValueError(f"No evaluation for frozen checkpoint at {episode} episodes")
        run_by_episode[episode] = eval_by_id[point["run_id"]]
    if full["run_id"] not in eval_by_id:
        raise ValueError("No evaluation for full-GRU 25k checkpoint")
    full_eval = eval_by_id[full["run_id"]]

    samples = {}
    checkpoint_process_wall = {}
    vi_reference = None
    square_reference = None
    for episode, run in run_by_episode.items():
        squares, gru, vi = rollout_samples(run)
        if square_reference is None:
            square_reference, vi_reference = squares, vi
        elif squares != square_reference or not np.array_equal(vi, vi_reference):
            raise ValueError("Frozen evaluations did not use common scenarios")
        samples[episode] = gru
        point = frozen_points[episode]
        point_metadata_path = (ROOT / point["checkpoint"]).with_suffix(".json")
        point_metadata = json.loads(
            point_metadata_path.read_text(encoding="utf-8")
        )
        started = datetime.fromisoformat(frozen["started_at_utc"])
        saved = datetime.fromisoformat(point_metadata["finished_at_utc"])
        checkpoint_process_wall[episode] = (saved - started).total_seconds()
    full_squares, full_samples, full_vi = rollout_samples(full_eval)
    if full_squares != square_reference or not np.array_equal(full_vi, vi_reference):
        raise ValueError("Full and frozen evaluations did not use common scenarios")

    plateau_delta = samples[25_000] - samples[20_000]
    early_quality_delta = samples[20_000] - full_samples
    same_budget_delta = samples[25_000] - full_samples
    plateau = ci95(plateau_delta)
    early_quality = ci95(early_quality_delta)
    same_budget_quality = ci95(same_budget_delta)

    full_ckpt = checkpoint(full["checkpoint"])
    pre_ckpt = checkpoint(frozen["pre_freeze_checkpoint"])
    prefix_match = torch.equal(
        full_ckpt["episode_returns"][:freeze_episode],
        pre_ckpt["episode_returns"],
    )
    pre_policy = pre_ckpt["policy_state_dict"]
    pre_target = pre_ckpt["target_state_dict"]
    gru_keys = [key for key in pre_policy if key.startswith("gru.")]
    q_keys = [key for key in pre_policy if key.startswith("q_head.")]
    integrity = {"episode_returns_match_through_freeze": bool(prefix_match),
                 "checkpoints": {}}
    for episode in wanted_episodes:
        ckpt = (checkpoint(frozen_points[episode]["checkpoint"])
                if episode != 25_000
                else checkpoint(frozen["checkpoint"]))
        online_fixed = bool(gru_keys) and all(
            torch.equal(pre_policy[key], ckpt["policy_state_dict"][key])
            for key in gru_keys
        )
        target_matches = bool(gru_keys) and all(
            torch.equal(pre_policy[key], ckpt["target_state_dict"][key])
            for key in gru_keys
        )
        q_changed = any(
            not torch.equal(pre_policy[key], ckpt["policy_state_dict"][key])
            for key in q_keys
        )
        integrity["checkpoints"][str(episode)] = {
            "online_gru_bitwise_unchanged": online_fixed,
            "target_gru_matches_frozen_gru": target_matches,
            "q_head_changed_from_freeze_point": bool(q_changed),
        }

    full_seconds = float(full["training_loop_seconds"])
    frozen20_seconds = float(frozen_points[20_000]["training_loop_seconds"])
    frozen25_seconds = float(frozen["training_loop_seconds"])
    result = {
        "design": "single-seed matched convergence demonstration; not population inference",
        "seed": int(full["training_seed"]),
        "freeze_after_episodes": freeze_episode,
        "evaluation_config": evaluation["evaluation_batch"],
        "evaluation_scenarios_per_policy": int(samples[20_000].size),
        "returns_by_checkpoint": {
            str(ep): ci95(values) for ep, values in samples.items()
        } | {"full_gru_25000": ci95(full_samples)},
        "paired_differences": {
            "frozen_25000_minus_frozen_20000": plateau,
            "frozen_20000_minus_full_25000": early_quality,
            "frozen_15000_minus_full_25000_exploratory": ci95(
                samples[15_000] - full_samples
            ),
            "frozen_25000_minus_full_25000": same_budget_quality,
        },
        "criteria": {
            "practical_plateau_tolerance": PLATEAU_TOLERANCE,
            "frozen_20k_to_25k_plateau_pass": bool(
                plateau["ci95"][0] >= -PLATEAU_TOLERANCE
                and plateau["ci95"][1] <= PLATEAU_TOLERANCE
            ),
            "quality_noninferiority_margin": QUALITY_MARGIN,
            "frozen_20k_vs_full_25k_pass": bool(
                early_quality["ci95"][0] > -QUALITY_MARGIN
            ),
        },
        "training_loop_seconds": {
            "full_gru_25000": full_seconds,
            "frozen_gru_20000": frozen20_seconds,
            "frozen_gru_25000": frozen25_seconds,
            "saved_at_frozen_20000_vs_full_25000": full_seconds - frozen20_seconds,
            "saved_at_frozen_25000_vs_full_25000": full_seconds - frozen25_seconds,
        },
        "process_wall_seconds": {
            "full_gru_25000": float(full["process_wall_seconds"]),
            "frozen_elapsed_to_checkpoints": {
                str(ep): seconds
                for ep, seconds in checkpoint_process_wall.items()
            },
            "savings_vs_full_gru_25000_by_frozen_checkpoint": {
                str(ep): {
                    "seconds": float(full["process_wall_seconds"]) - seconds,
                    "fraction": (
                        float(full["process_wall_seconds"]) - seconds
                    ) / float(full["process_wall_seconds"]),
                }
                for ep, seconds in checkpoint_process_wall.items()
            },
            "frozen_gru_25000": float(frozen["process_wall_seconds"]),
            "saved_at_equal_25000_budget": (
                float(full["process_wall_seconds"])
                - float(frozen["process_wall_seconds"])
            ),
        },
        "integrity": integrity,
        "scope_note": (
            "All confidence intervals describe rollout scenarios for one training seed. "
            "They do not quantify seed-to-seed variation."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    episodes = [10_000, 15_000, 20_000, 25_000]
    means = [float(np.mean(samples[ep])) for ep in episodes]
    errors = [
        (ci95(samples[ep])["ci95"][1] - ci95(samples[ep])["ci95"][0]) / 2
        for ep in episodes
    ]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(np.asarray(episodes) / 1000, means, yerr=errors,
                marker="o", capsize=4, linewidth=2, label="GRU frozen after 10k")
    full_summary = ci95(full_samples)
    ax.errorbar([25], [full_summary["mean"]],
                yerr=[[(full_summary["mean"] - full_summary["ci95"][0])],
                      [(full_summary["ci95"][1] - full_summary["mean"])]],
                marker="s", capsize=4, linestyle="none", label="Full GRU 25k")
    ax.axvspan(20, 25, color="grey", alpha=.12, label="Plateau check")
    ax.set(xlabel="Training episodes (thousands)", ylabel="Mean discounted return",
           title="Head-only policy trajectory and full-GRU reference")
    ax.grid(alpha=.25)
    ax.legend()
    fig.tight_layout()
    args.plot.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.plot, dpi=180)
    plt.close(fig)
    print(json.dumps({
        "plateau": plateau,
        "frozen20_minus_full25": early_quality,
        "training_loop_saved_seconds_at_20k": full_seconds - frozen20_seconds,
        "integrity": integrity,
        "output": str(args.output),
        "plot": str(args.plot),
    }, indent=2))


if __name__ == "__main__":
    main()
