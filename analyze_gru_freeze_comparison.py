"""Summarize paired return and wall-time results for GRU freezing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch


ROOT = Path(__file__).resolve().parent


T_CRITICAL_95 = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776,
                 6: 2.571, 7: 2.447, 8: 2.365, 9: 2.306, 10: 2.262,
                 11: 2.228, 12: 2.201, 13: 2.179, 14: 2.160,
                 15: 2.145, 16: 2.131, 17: 2.120, 18: 2.110,
                 19: 2.101, 20: 2.093, 21: 2.086, 22: 2.080,
                 23: 2.074, 24: 2.069, 25: 2.064, 26: 2.060,
                 27: 2.056, 28: 2.052, 29: 2.045, 30: 2.042}
NONINFERIORITY_MARGIN = 0.05


def mean_ci(values, critical=1.96):
    x = np.asarray(values, dtype=float)
    mean = float(x.mean())
    se = float(x.std(ddof=1) / np.sqrt(x.size)) if x.size > 1 else 0.0
    return {"mean": mean, "ci95": [mean - critical * se, mean + critical * se],
            "n": int(x.size)}


def across_seed_ci(values):
    x = np.asarray(values, dtype=float)
    n = int(x.size)
    critical = T_CRITICAL_95.get(n, 1.96)
    summary = mean_ci(x, critical)
    summary["n_training_seeds"] = n
    return summary


def rollout_samples(run):
    squares = [item["square"] for item in run["starts"]]
    gru = np.asarray([
        value
        for item in run["starts"]
        for value in item["rollout_samples"]["gru_dqn"]
    ], dtype=float)
    vi = np.asarray([
        value
        for item in run["starts"]
        for value in item["rollout_samples"][
            "mdp_value_iteration_on_noisy_observation"
        ]
    ], dtype=float)
    return squares, gru, vi


def paired_summary(samples):
    x = np.asarray(samples, dtype=float)
    return mean_ci(x, 1.96)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("evaluation", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plot", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    evaluation = json.loads(args.evaluation.read_text(encoding="utf-8"))
    freeze_episode = int(manifest["config"]["freeze_after_episodes"])
    runs = [item for item in manifest["runs"] if item["status"] == "complete"]
    expected = len(manifest["config"]["seeds"]) * 2
    if len(runs) != expected:
        raise ValueError(f"Expected {expected} training runs, found {len(runs)}")
    eval_by_id = {item["run_id"]: item for item in evaluation["runs"]}
    training_by_seed = {}
    for item in runs:
        training_by_seed.setdefault(int(item["training_seed"]), {})[item["arm"]] = item

    records = []
    for seed in sorted(training_by_seed):
        arms = training_by_seed[seed]
        if set(arms) != {"full", "frozen"}:
            raise ValueError(f"Seed {seed} does not have both training arms")
        full_record = arms["full"]
        frozen_record = arms["frozen"]
        full = eval_by_id[full_record["run_id"]]
        frozen = eval_by_id[frozen_record["run_id"]]
        pre_freeze = eval_by_id[frozen_record["pre_freeze_run_id"]]
        if not (full["evaluation_seed"] == frozen["evaluation_seed"]
                == pre_freeze["evaluation_seed"]):
            raise ValueError(f"Evaluation seed mismatch for training seed {seed}")
        squares_full, full_samples, vi_samples = rollout_samples(full)
        squares_frozen, frozen_samples, _ = rollout_samples(frozen)
        squares_pre, pre_samples, _ = rollout_samples(pre_freeze)
        if squares_full != squares_frozen or squares_full != squares_pre:
            raise ValueError(f"Start squares differ for training seed {seed}")
        if not (len(full_samples) == len(frozen_samples) == len(pre_samples)):
            raise ValueError(f"Rollout counts differ for training seed {seed}")
        if not np.array_equal(vi_samples, rollout_samples(frozen)[2]):
            raise ValueError(f"VI scenarios differ for training seed {seed}")

        full_checkpoint = torch.load(
            ROOT / full_record["checkpoint"], map_location="cpu", weights_only=True
        )
        frozen_checkpoint = torch.load(
            ROOT / frozen_record["checkpoint"], map_location="cpu", weights_only=True
        )
        pre_checkpoint = torch.load(
            ROOT / frozen_record["pre_freeze_checkpoint"],
            map_location="cpu", weights_only=True,
        )
        prefix_matches = torch.equal(
            full_checkpoint["episode_returns"][:pre_checkpoint["episode_returns"].numel()],
            pre_checkpoint["episode_returns"],
        )
        gru_keys = [
            key for key in pre_checkpoint["policy_state_dict"]
            if key.startswith("gru.")
        ]
        gru_frozen_exactly = bool(gru_keys) and all(
            torch.equal(
                pre_checkpoint["policy_state_dict"][key],
                frozen_checkpoint["policy_state_dict"][key],
            )
            for key in gru_keys
        )
        target_gru_keys = [
            key for key in pre_checkpoint["target_state_dict"]
            if key.startswith("gru.")
        ]
        target_gru_matches_frozen_policy = bool(target_gru_keys) and all(
            torch.equal(
                frozen_checkpoint["target_state_dict"][key],
                frozen_checkpoint["policy_state_dict"][key],
            )
            for key in target_gru_keys
        )
        q_head_changed = any(
            not torch.equal(
                pre_checkpoint["policy_state_dict"][key],
                frozen_checkpoint["policy_state_dict"][key],
            )
            for key in pre_checkpoint["policy_state_dict"]
            if key.startswith("q_head.")
        )
        if not prefix_matches:
            raise ValueError(f"Training prefixes do not match for seed {seed}")
        if not gru_frozen_exactly:
            raise ValueError(f"GRU weights changed after freezing for seed {seed}")
        if not target_gru_matches_frozen_policy:
            raise ValueError(
                f"Target GRU is not held at the frozen policy weights for seed {seed}"
            )
        if not q_head_changed:
            raise ValueError(f"Q head did not change after freezing for seed {seed}")

        progress = []
        quality_trajectory = [{
            "episodes": int(pre_checkpoint["metadata"]["episodes"]),
            "phase": "at_freeze_point",
            "mean_return": float(pre_samples.mean()),
            "paired_difference_from_end_to_end_final": float(
                (pre_samples - full_samples).mean()
            ),
            "training_loop_seconds": float(
                pre_checkpoint["metadata"]["training_loop_seconds"]
            ),
        }]
        progress_gru_checks = []
        target_gru_progress_checks = []
        for milestone in frozen_record.get("head_only_progress_checkpoints", []):
            milestone_metadata = json.loads(
                (ROOT / milestone["metadata"]).read_text(encoding="utf-8")
            )
            milestone_loop_seconds = float(
                milestone_metadata["training_loop_seconds"]
            )
            progress_checkpoint = torch.load(
                ROOT / milestone["checkpoint"],
                map_location="cpu", weights_only=True,
            )
            checkpoint_returns = progress_checkpoint["episode_returns"]
            if checkpoint_returns.numel() != int(milestone["episodes"]):
                raise ValueError(
                    f"Episode-return count mismatch at episode "
                    f"{milestone['episodes']} for seed {seed}"
                )
            shared_prefix_matches = (
                checkpoint_returns.numel() >= freeze_episode
                and torch.equal(
                    checkpoint_returns[:freeze_episode],
                    full_checkpoint["episode_returns"][:freeze_episode],
                )
                and torch.equal(
                    checkpoint_returns[:freeze_episode],
                    pre_checkpoint["episode_returns"],
                )
            )
            if not shared_prefix_matches:
                raise ValueError(
                    f"Training prefixes differ through freeze episode "
                    f"{freeze_episode} at checkpoint {milestone['episodes']} "
                    f"for seed {seed}"
                )
            progress_gru_frozen = bool(gru_keys) and all(
                torch.equal(
                    pre_checkpoint["policy_state_dict"][key],
                    progress_checkpoint["policy_state_dict"][key],
                )
                for key in gru_keys
            )
            progress_q_head_changed = any(
                not torch.equal(
                    pre_checkpoint["policy_state_dict"][key],
                    progress_checkpoint["policy_state_dict"][key],
                )
                for key in pre_checkpoint["policy_state_dict"]
                if key.startswith("q_head.")
            )
            if not progress_gru_frozen:
                raise ValueError(
                    f"GRU weights changed by episode {milestone['episodes']} "
                    f"for seed {seed}"
                )
            if not progress_q_head_changed:
                raise ValueError(
                    f"Q head did not change by episode {milestone['episodes']} "
                    f"for seed {seed}"
                )
            progress_target_gru_matches_policy = bool(target_gru_keys) and all(
                torch.equal(
                    progress_checkpoint["target_state_dict"][key],
                    pre_checkpoint["policy_state_dict"][key],
                )
                for key in target_gru_keys
            )
            if not progress_target_gru_matches_policy:
                raise ValueError(
                    f"Target GRU changed after freeze boundary by episode "
                    f"{milestone['episodes']} for seed {seed}"
                )
            progress_gru_checks.append({
                "episodes": int(milestone["episodes"]),
                "shared_prefix_matches_full_arm_through_freeze": True,
                "shared_prefix_episodes": freeze_episode,
                "gru_weights_bitwise_unchanged": True,
                "q_head_weights_changed": True,
            })
            target_gru_progress_checks.append({
                "episodes": int(milestone["episodes"]),
                "target_gru_matches_frozen_policy_gru": True,
            })
            progress_run = eval_by_id[milestone["run_id"]]
            progress_squares, progress_samples, progress_vi = rollout_samples(progress_run)
            if progress_squares != squares_full or not np.array_equal(vi_samples, progress_vi):
                raise ValueError(
                    f"Evaluation scenarios differ at episode {milestone['episodes']} for seed {seed}"
                )
            progress.append({
                "episodes": int(milestone["episodes"]),
                "training_loop_seconds": milestone_loop_seconds,
                "mean_return": float(progress_samples.mean()),
                "paired_difference_from_pre_freeze": {
                    "mean": float((progress_samples - pre_samples).mean()),
                    "rollout_ci95": paired_summary(progress_samples - pre_samples)["ci95"],
                },
            })
            quality_trajectory.append({
                "episodes": int(milestone["episodes"]),
                "phase": "head_only",
                "mean_return": float(progress_samples.mean()),
                "paired_difference_from_end_to_end_final": float(
                    (progress_samples - full_samples).mean()
                ),
                "training_loop_seconds": milestone_loop_seconds,
            })

        frozen_minus_full = frozen_samples - full_samples
        frozen_minus_pre = frozen_samples - pre_samples
        full_seconds = float(full_record["process_wall_seconds"])
        frozen_seconds = float(frozen_record["process_wall_seconds"])
        full_loop_seconds = float(full_record["training_loop_seconds"])
        saved_seconds = full_seconds - frozen_seconds
        saved_fraction = saved_seconds / full_seconds
        frozen_final_episodes = int(frozen_checkpoint["metadata"]["episodes"])
        quality_trajectory.append({
            "episodes": frozen_final_episodes,
            "phase": "frozen_final",
            "mean_return": float(frozen_samples.mean()),
            "paired_difference_from_end_to_end_final": float(
                frozen_minus_full.mean()
            ),
            "training_loop_seconds": float(
                frozen_record["training_loop_seconds"]
            ),
        })
        for point in quality_trajectory:
            point["trainer_loop_seconds_saved_vs_full_final"] = (
                full_loop_seconds - point["training_loop_seconds"]
            )
        records.append({
            "training_seed": seed,
            "scenario_count": int(len(full_samples)),
            "treatment_integrity": {
                "shared_training_prefix_matches_full_arm_through_freeze": prefix_matches,
                "gru_weights_bitwise_unchanged_after_freeze": gru_frozen_exactly,
                "target_gru_synced_at_freeze_and_unchanged": (
                    target_gru_matches_frozen_policy
                    and all(target_gru_progress_checks)
                ),
                "q_head_weights_changed_after_freeze": q_head_changed,
                "head_only_checkpoint_checks": progress_gru_checks,
                "target_gru_checkpoint_checks": target_gru_progress_checks,
            },
            "mean_return": {
                "pre_freeze": float(pre_samples.mean()),
                "frozen_final": float(frozen_samples.mean()),
                "end_to_end_final": float(full_samples.mean()),
                "noisy_input_value_iteration": float(vi_samples.mean()),
            },
            "frozen_minus_end_to_end": {
                "mean": float(frozen_minus_full.mean()),
                "rollout_ci95": paired_summary(frozen_minus_full)["ci95"],
            },
            "frozen_final_minus_pre_freeze": {
                "mean": float(frozen_minus_pre.mean()),
                "rollout_ci95": paired_summary(frozen_minus_pre)["ci95"],
            },
            "head_only_learning_curve": progress,
            "frozen_policy_quality_trajectory": quality_trajectory,
            "wall_time": {
                "full_process_seconds": full_seconds,
                "frozen_process_seconds": frozen_seconds,
                "seconds_saved": saved_seconds,
                "fraction_saved": saved_fraction,
                "joint_phase_seconds": float(
                    frozen_record["joint_gru_head_training_seconds"]
                ),
                "head_only_phase_seconds": float(
                    frozen_record["head_only_training_seconds"]
                ),
            },
        })

    frozen_full_delta = [r["frozen_minus_end_to_end"]["mean"] for r in records]
    frozen_pre_delta = [r["frozen_final_minus_pre_freeze"]["mean"] for r in records]
    saved_seconds = [r["wall_time"]["seconds_saved"] for r in records]
    saved_fraction = [r["wall_time"]["fraction_saved"] for r in records]
    full_means = [r["mean_return"]["end_to_end_final"] for r in records]
    frozen_means = [r["mean_return"]["frozen_final"] for r in records]
    pre_means = [r["mean_return"]["pre_freeze"] for r in records]
    vi_means = [r["mean_return"]["noisy_input_value_iteration"] for r in records]

    shared_milestones = sorted(set.intersection(*[
        {point["episodes"] for point in record["head_only_learning_curve"]}
        for record in records
    ])) if records else []
    curve_summary = []
    for episode in shared_milestones:
        values = []
        deltas = []
        times = []
        for record in records:
            point = next(p for p in record["head_only_learning_curve"]
                         if p["episodes"] == episode)
            values.append(point["mean_return"])
            deltas.append(point["paired_difference_from_pre_freeze"]["mean"])
            times.append(point["training_loop_seconds"])
        curve_summary.append({
            "episodes": episode,
            "mean_return_across_seeds": across_seed_ci(values),
            "paired_change_from_freeze_point": across_seed_ci(deltas),
            "mean_training_loop_seconds": float(np.mean(times)),
        })

    shared_quality_episodes = sorted(set.intersection(*[
        {point["episodes"] for point in record["frozen_policy_quality_trajectory"]}
        for record in records
    ])) if records else []
    quality_trajectory_summary = []
    for episode in shared_quality_episodes:
        points = [
            next(point for point in record["frozen_policy_quality_trajectory"]
                 if point["episodes"] == episode)
            for record in records
        ]
        difference = across_seed_ci([
            point["paired_difference_from_end_to_end_final"]
            for point in points
        ])
        time_saved = across_seed_ci([
            point["trainer_loop_seconds_saved_vs_full_final"]
            for point in points
        ])
        quality_trajectory_summary.append({
            "episodes": episode,
            "phase": points[0]["phase"],
            "mean_return_across_seeds": across_seed_ci([
                point["mean_return"] for point in points
            ]),
            "paired_difference_from_end_to_end_final": difference,
            "trainer_loop_seconds_saved_vs_full_final": time_saved,
            "within_quality_margin_at_this_checkpoint": (
                difference["ci95"][0] > -NONINFERIORITY_MARGIN
            ),
            "positive_trainer_time_saving_at_this_checkpoint": (
                time_saved["ci95"][0] > 0.0
            ),
        })

    quality_ci = across_seed_ci(frozen_full_delta)
    noninferiority_passed = quality_ci["ci95"][0] > -NONINFERIORITY_MARGIN
    time_ci = across_seed_ci(saved_seconds)
    summary = {
        "question": (
            f"Does freezing the history-4 GRU at episode "
            f"{manifest['config']['freeze_after_episodes']:,} and training only "
            "its Q head preserve policy quality while saving training time?"
        ),
        "training_config": manifest["config"],
        "evaluation_config": evaluation["evaluation_batch"],
        "registered_noninferiority_margin": NONINFERIORITY_MARGIN,
        "milestone_time_to_quality_analysis": (
            "Exploratory secondary analysis; the registered primary quality "
            "comparison is the frozen-final checkpoint versus end-to-end-final."
        ),
        "n_training_seeds": len(records),
        "returns_across_training_seeds": {
            "pre_freeze": across_seed_ci(pre_means),
            "frozen_final": across_seed_ci(frozen_means),
            "end_to_end_final": across_seed_ci(full_means),
            "noisy_input_value_iteration": across_seed_ci(vi_means),
        },
        "frozen_minus_end_to_end": quality_ci,
        "frozen_final_minus_pre_freeze": across_seed_ci(frozen_pre_delta),
        "head_only_learning_curve": curve_summary,
        "frozen_policy_quality_trajectory": quality_trajectory_summary,
        "seconds_saved_full_minus_frozen": time_ci,
        "fraction_wall_time_saved": across_seed_ci(saved_fraction),
        "noninferiority_rule": {
            "margin": NONINFERIORITY_MARGIN,
            "lower_bound_exceeds_minus_margin": noninferiority_passed,
        },
        "wall_time_rule": {
            "lower_ci_bound_exceeds_zero": time_ci["ci95"][0] > 0.0,
        },
        "interpretation": (
            "Both registered criteria passed."
            if noninferiority_passed
            and time_ci["ci95"][0] > 0.0 else
            "At least one registered criterion did not pass; do not claim that freezing both preserves quality and saves time."
        ),
        "per_seed": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6))
    milestone_points = [
        {"episodes": manifest["config"]["freeze_after_episodes"],
         "key": "pre_freeze", "label": "At freeze point"}
    ] + [
        {"episodes": point["episodes"], "key": None,
         "label": f"Frozen head at {point['episodes'] // 1000}k"}
        for point in curve_summary
    ] + [{
        "episodes": manifest["config"]["episodes"],
        "key": "frozen_final", "label": "Frozen head final",
    }, {
        "episodes": manifest["config"]["episodes"],
        "key": "end_to_end_final", "label": "End-to-end final",
    }]
    for record in records:
        seed_x, seed_y = [], []
        for point_index, point in enumerate(milestone_points):
            if point["key"] is not None:
                seed_x.append(point_index)
                seed_y.append(record["mean_return"][point["key"]])
            else:
                checkpoint = next(
                    p for p in record["head_only_learning_curve"]
                    if p["episodes"] == point["episodes"]
                )
                seed_x.append(point_index)
                seed_y.append(checkpoint["mean_return"])
        axes[0].plot(seed_x, seed_y, marker="o", alpha=.45,
                     linewidth=.8, label=f"seed {record['training_seed']}")
    aggregate_returns = []
    for point in milestone_points:
        if point["key"] is not None:
            aggregate_returns.append([
                record["mean_return"][point["key"]] for record in records
            ])
        else:
            aggregate_returns.append([
                next(p["mean_return"] for p in record["head_only_learning_curve"]
                     if p["episodes"] == point["episodes"])
                for record in records
            ])
    for point_index, values in enumerate(aggregate_returns):
        stats = across_seed_ci(values)
        mean = stats["mean"]
        lo, hi = stats["ci95"]
        axes[0].errorbar(point_index, mean, yerr=[[mean - lo], [hi - mean]],
                         fmt="D", color="#171717", capsize=3,
                         markersize=4, zorder=10)
    average_vi = float(np.mean(vi_means))
    axes[0].axhline(average_vi, color="#d95f0e", linestyle="--",
                    label="Noisy-input MDP value iteration")
    axes[0].set_xticks(range(len(milestone_points)),
                       [p["label"] for p in milestone_points],
                       rotation=20, ha="right")
    axes[0].set_xlabel("Training checkpoint")
    axes[0].set_ylabel("Mean discounted return")
    axes[0].set_title("Greedy return as only the Q head continues training")
    axes[0].legend(fontsize=7, ncol=2)
    axes[0].grid(axis="y", alpha=.25)

    savings_pct = np.asarray(saved_fraction) * 100
    x = np.arange(len(records))
    mean_pct = float(savings_pct.mean())
    se_pct = float(savings_pct.std(ddof=1) / np.sqrt(len(savings_pct)))
    tcrit = T_CRITICAL_95.get(len(savings_pct), 1.96)
    axes[1].bar(x, savings_pct, color="#756bb1")
    axes[1].axhline(mean_pct, color="#252525", linewidth=1.3,
                    label=f"Mean {mean_pct:.1f}%")
    axes[1].fill_between(
        [-.5, len(records) - .5],
        mean_pct - tcrit * se_pct,
        mean_pct + tcrit * se_pct,
        color="#756bb1", alpha=.15, label="95% CI for mean",
    )
    axes[1].axhline(0, color="#444", linewidth=.8)
    axes[1].set_xticks(x, [str(r["training_seed"]) for r in records])
    axes[1].set_xlabel("Training seed")
    axes[1].set_ylabel("Process wall time saved (%)")
    axes[1].set_title("Frozen arm versus end-to-end training")
    axes[1].legend(fontsize=8)
    axes[1].grid(axis="y", alpha=.25)
    fig.suptitle(
        f"Freeze the GRU after {manifest['config']['freeze_after_episodes']:,} "
        "episodes; train only the Q head"
    )
    fig.tight_layout()
    args.plot.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.plot, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(json.dumps({
        "frozen_minus_end_to_end": summary["frozen_minus_end_to_end"],
        "frozen_final_minus_pre_freeze": summary["frozen_final_minus_pre_freeze"],
        "seconds_saved_full_minus_frozen": summary["seconds_saved_full_minus_frozen"],
        "fraction_wall_time_saved": summary["fraction_wall_time_saved"],
        "interpretation": summary["interpretation"],
    }, indent=2))
    print(f"Saved analysis {args.output}")
    print(f"Saved figure {args.plot}")


if __name__ == "__main__":
    main()
