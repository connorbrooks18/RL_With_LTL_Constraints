"""Evaluate q-observed GRU, frozen-GRU, and RSSM checkpoints over time."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import time

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

from dqn import GRUDQN, RSSMDQN
from evaluate_pomdp import build_environment
from evaluate_pomdp_vs_value_iteration import _rollout


T95 = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571,
       7: 2.447, 8: 2.365, 9: 2.306, 10: 2.262}


def interval(values):
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    if values.size < 2:
        half = 0.0
    else:
        half = T95.get(values.size, 1.96) * float(values.std(ddof=1)) / np.sqrt(values.size)
    return {"mean": mean, "ci95_across_training_seeds": [mean - half, mean + half],
            "n_training_seeds": int(values.size)}


def load_manifest(path: Path):
    manifest = json.loads(path.read_text(encoding="utf-8"))
    checkpoint_records = []
    for run in manifest.get("runs", []):
        if run.get("status") != "complete" or run.get("arm") not in ("gru", "frozen", "rssm"):
            continue
        checkpoints = list(run.get("progress_checkpoints", []))
        checkpoints.append({
            "episodes": int(run["episodes"]),
            "checkpoint": run["checkpoint"],
            "training_loop_seconds": run["training_loop_seconds"],
        })
        for item in checkpoints:
            checkpoint_records.append({
                "training_seed": int(run["training_seed"]),
                "arm": run["arm"],
                **item,
            })
    return manifest, checkpoint_records


def load_model(path: Path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    metadata = checkpoint["metadata"]
    if metadata.get("architecture") != "rssm":
        model = GRUDQN(
            int(metadata["n_observations"]), int(metadata["n_actions"]),
            hidden_size=int(metadata["gru_hidden_size"]),
            num_layers=int(metadata["gru_num_layers"]),
        )
    else:
        _grid, _oa, csrl, *_ = build_environment(float(metadata["observation_error"]))
        factors = (
            tuple(csrl.shape[:-1]) if metadata.get("automaton_state_observed", False)
            else (csrl.shape[0], csrl.mdp.shape[0], csrl.mdp.shape[1])
        )
        model = RSSMDQN(
            int(metadata["n_observations"]), int(metadata["n_actions"]), factors,
            hidden_size=int(metadata["rssm_deterministic_size"]),
            stochastic_variables=int(metadata["rssm_stochastic_variables"]),
            categorical_classes=int(metadata["rssm_categorical_classes"]),
        )
    model.load_state_dict(checkpoint["policy_state_dict"])
    model.eval()
    return model, metadata


def evaluate_one(checkpoint_record, *, runs: int, horizon: int,
                 evaluation_seed: int, include_vi: bool = False):
    evaluation_started = time.perf_counter()
    path = Path(checkpoint_record["checkpoint"])
    if not path.is_absolute():
        path = Path(__file__).resolve().parent / path
    model, metadata = load_model(path)
    if not metadata.get("partial_observation") or not metadata.get("automaton_state_observed"):
        raise ValueError(f"Checkpoint is not q-observed with noisy grid position: {path}")
    grid, _oa, csrl, starts, *_ = build_environment(float(metadata["observation_error"]))
    if include_vi:
        vi_value = csrl.value_iteration(threshold=1e-8)
        vi_policy = csrl.greedy_policy(vi_value)
    else:
        vi_value = vi_policy = None
    results = []
    for start_index, start in enumerate(starts):
        samples = []
        vi_samples = []
        for rollout_index in range(runs):
            rollout_seed = evaluation_seed + start_index * 1_000_003 + rollout_index
            samples.append(_rollout(
                csrl, start, rollout_seed, horizon,
                model=model, metadata=metadata,
            ))
            if include_vi:
                vi_samples.append(_rollout(
                    csrl, start, rollout_seed, horizon,
                    value_iteration_value=vi_value,
                    value_iteration_policy=vi_policy,
                    automaton_state_observed=True,
                ))
        start_record = {
            "square": [int(value) for value in start],
            "label": list(grid.label[start]),
            "returns": samples,
            "mean": float(np.mean(samples)),
        }
        if include_vi:
            start_record.update({
                "vi_returns": vi_samples,
                "vi_mean": float(np.mean(vi_samples)),
                "paired_difference_vs_vi": float(
                    np.mean(np.asarray(samples) - np.asarray(vi_samples))
                ),
            })
        results.append(start_record)
    flat = [value for start in results for value in start["returns"]]
    output = {
        **checkpoint_record,
        "checkpoint": str(path),
        "automaton_state_observed": True,
        "grid_position_partially_observed": True,
        "training_loop_seconds": float(
            metadata.get("training_loop_seconds", checkpoint_record["training_loop_seconds"])
        ),
        "rollouts_per_start": runs,
        "horizon": horizon,
        "evaluation_wall_seconds": time.perf_counter() - evaluation_started,
        "mean_return": float(np.mean(flat)),
        "start_results": results,
    }
    if include_vi:
        vi_flat = [value for start in results for value in start["vi_returns"]]
        output["mean_vi_return"] = float(np.mean(vi_flat))
        output["paired_difference_vs_vi"] = float(
            np.mean(np.asarray(flat) - np.asarray(vi_flat))
        )
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path,
        default=Path("artifacts/q_observed_section_one/training_manifest.json"),
    )
    parser.add_argument("--runs", type=int, default=20,
                        help="common rollouts per traversable start")
    parser.add_argument("--horizon", type=int, default=500)
    parser.add_argument("--episodes", type=int, nargs="+",
                        default=(10_000, 12_500, 15_000),
                        help="checkpoint budgets to evaluate")
    parser.add_argument("--include-vi", action="store_true",
                        help="also evaluate q-aware noisy-input MDP value iteration")
    parser.add_argument("--seed", type=int, default=20261110)
    parser.add_argument("--output-dir", type=Path,
                        default=Path("artifacts/q_observed_section_one/convergence"))
    args = parser.parse_args()
    if args.runs < 1 or args.horizon < 1:
        parser.error("--runs and --horizon must be positive")
    torch.set_num_threads(1)
    manifest, checkpoints = load_manifest(args.manifest)
    wanted_episodes = set(map(int, args.episodes))
    checkpoints = [item for item in checkpoints
                   if int(item["episodes"]) in wanted_episodes]
    expected_seeds = set(map(int, manifest["config"].get("seeds", ())))
    for arm in ("gru", "frozen", "rssm"):
        for episodes in sorted(wanted_episodes):
            found = {
                int(item["training_seed"]) for item in checkpoints
                if item["arm"] == arm and int(item["episodes"]) == episodes
            }
            if found != expected_seeds:
                missing = sorted(expected_seeds - found)
                raise SystemExit(
                    f"Need complete {arm} checkpoints at {episodes} episodes; "
                    f"missing seeds {missing}"
                )
    records = []
    for index, checkpoint in enumerate(checkpoints, 1):
        print(
            f"[{index}/{len(checkpoints)}] seed {checkpoint['training_seed']} "
            f"{checkpoint['arm']} K={checkpoint['episodes']}", flush=True,
        )
        records.append(evaluate_one(
            checkpoint, runs=args.runs, horizon=args.horizon,
            evaluation_seed=args.seed, include_vi=args.include_vi,
        ))

    # Per-checkpoint paired changes against full GRU use the same training seed,
    # start squares, rollout seeds, and horizon.
    by_key = {
        (record["training_seed"], record["arm"], record["episodes"]): record
        for record in records
    }
    for record in records:
        if record["arm"] == "gru":
            record["paired_delta_vs_full_gru"] = 0.0
            continue
        reference = by_key.get((record["training_seed"], "gru", record["episodes"]))
        if reference is not None:
            record["paired_delta_vs_full_gru"] = float(
                np.mean([
                    candidate - base
                    for candidate_start, base_start in zip(
                        record["start_results"], reference["start_results"]
                    )
                    for candidate, base in zip(
                        candidate_start["returns"], base_start["returns"]
                    )
                ])
            )

    aggregate = {}
    for arm in ("gru", "frozen", "rssm"):
        for episodes in sorted(wanted_episodes):
            group = [record for record in records
                     if record["arm"] == arm and int(record["episodes"]) == episodes]
            aggregate[f"{arm}_K{episodes}"] = {
                "return": interval([item["mean_return"] for item in group]),
                "training_loop_seconds": interval([
                    item["training_loop_seconds"] for item in group
                ]),
                "paired_time_saving_vs_full_gru_seconds": interval([
                    by_key[(int(item["training_seed"]), "gru", episodes)][
                        "training_loop_seconds"
                    ] - item["training_loop_seconds"]
                    for item in group
                ]),
                "paired_delta_vs_full_gru": (
                    interval([item["paired_delta_vs_full_gru"] for item in group])
                    if group and all("paired_delta_vs_full_gru" in item for item in group)
                    else None
                ),
                "vi_return": (
                    interval([item["mean_vi_return"] for item in group])
                    if group and all("mean_vi_return" in item for item in group)
                    else None
                ),
                "paired_difference_vs_vi": (
                    interval([item["paired_difference_vs_vi"] for item in group])
                    if group and all("paired_difference_vs_vi" in item for item in group)
                    else None
                ),
            }

    time_to_threshold = {}
    if {10_000, 12_500, 15_000}.issubset(wanted_episodes):
        for arm in ("gru", "frozen", "rssm"):
            reached_seconds = []
            not_reached = []
            for training_seed in sorted(expected_seeds):
                final_reference = by_key[(training_seed, "gru", 15_000)]["mean_return"]
                threshold = 0.9 * final_reference
                candidates = [by_key[(training_seed, arm, episodes)]
                              for episodes in (10_000, 12_500, 15_000)
                              if by_key[(training_seed, arm, episodes)]["mean_return"]
                              >= threshold]
                if candidates:
                    first = min(candidates, key=lambda item: int(item["episodes"]))
                    reached_seconds.append(float(first["training_loop_seconds"]))
                else:
                    not_reached.append(training_seed)
            time_to_threshold[arm] = {
                "threshold_rule": "90% of the same seed's full-GRU 15k return",
                "training_loop_seconds_for_reached_seeds": (
                    interval(reached_seconds) if reached_seconds else None
                ),
                "n_reached": len(reached_seconds),
                "n_training_seeds": len(expected_seeds),
                "not_reached_seeds": not_reached,
            }

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "config": {
            "training_seeds": manifest["config"].get("seeds"),
            "evaluation_seed": args.seed,
            "rollouts_per_start": args.runs,
            "horizon": args.horizon,
            "grid_position_partially_observed": True,
            "automaton_state_observed": True,
            "metric": "mean discounted return",
            "include_q_aware_noisy_input_mdp_value_iteration": bool(args.include_vi),
        },
        "aggregate": aggregate,
        "time_to_threshold": time_to_threshold,
        "checkpoints": records,
    }
    json_path = output_dir / "evaluation.json"
    json_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    with (output_dir / "evaluation.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=(
            "training_seed", "arm", "episodes", "training_loop_seconds",
            "mean_return", "paired_delta_vs_full_gru", "mean_vi_return",
            "paired_difference_vs_vi", "checkpoint",
        ))
        writer.writeheader()
        for record in records:
            writer.writerow({key: record.get(key) for key in writer.fieldnames})

    grid, *_ = build_environment(0.25)
    for episodes in sorted(wanted_episodes):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5.5), constrained_layout=True)
        groups_by_arm = {
            arm: [record for record in records
                  if record["arm"] == arm and int(record["episodes"]) == episodes]
            for arm in ("gru", "frozen", "rssm")
        }
        grid_maps = {}
        for arm, group in groups_by_arm.items():
            grid_values = np.zeros(grid.shape, dtype=float)
            for item in group:
                for start in item["start_results"]:
                    row, col = start["square"]
                    grid_values[row, col] += start["mean"] / len(group)
            grid_maps[arm] = grid_values
        shared_vmax = max((
            float(np.nanmax(values)) for values in grid_maps.values() if values.size
        ), default=0.01)
        shared_vmax = max(0.01, shared_vmax)
        for axis, arm in zip(axes, ("gru", "frozen", "rssm")):
            group = groups_by_arm[arm]
            grid_values = grid_maps[arm]
            if group:
                cmap = grid.cmap.copy()
                cmap.set_bad("#777777")
                shown_values = np.ma.masked_where(grid.structure == "B", grid_values)
                image = axis.imshow(
                    shown_values, interpolation="nearest", cmap=cmap,
                    vmin=0.0, vmax=shared_vmax,
                )
                fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04,
                             label="Mean discounted return")
                axis.set_title(f"{arm.upper()} · {episodes:,} episodes")
                axis.set_xticks(np.arange(grid.shape[1]))
                axis.set_yticks(np.arange(grid.shape[0]))
                axis.set_xticks(np.arange(-.5, grid.shape[1], 1), minor=True)
                axis.set_yticks(np.arange(-.5, grid.shape[0], 1), minor=True)
                axis.xaxis.tick_top()
                axis.grid(which="minor", color="lightgray", linewidth=0.8, alpha=0.7)
                axis.tick_params(which="minor", bottom=False, left=False)
                axis.set_xlim(-.5, grid.shape[1] - .5)
                axis.set_ylim(grid.shape[0] - .5, -.5)
                for row, col in grid.states():
                    if grid.structure[row, col] == "B":
                        axis.text(col, row, "B", ha="center", va="center",
                                  color="white", fontweight="bold")
                        continue
                    if grid.structure[row, col] == "T":
                        axis.add_patch(plt.Circle((col, row), 0.45,
                                                  fill=False, color="black", lw=1.2))
                    labels = grid.label[row, col]
                    if labels:
                        axis.text(col, row - 0.2, ",".join(labels),
                                  ha="center", va="center", fontsize=8,
                                  fontweight="bold")
                    axis.text(col, row + 0.19, f"{grid_values[row, col]:.2f}",
                              ha="center", va="center", fontsize=8)
            else:
                axis.set_title(f"{arm.upper()} · pending")
        fig.suptitle(
            f"q observed; noisy grid position · paired mean return by start · {episodes:,} episodes"
        )
        fig.savefig(output_dir / f"grid_returns_{episodes}.png", dpi=170)
        plt.close(fig)
        notebook_grid_dir = output_dir / "notebook_grid_returns"
        notebook_grid_dir.mkdir(parents=True, exist_ok=True)
        for arm, group in groups_by_arm.items():
            if not group:
                continue
            grid.plot(
                value=grid_maps[arm],
                save=str(notebook_grid_dir / f"{arm}_K{episodes}.png"),
            )
            plt.close(plt.gcf())

    for xkey, xlabel, filename in (
        ("episodes", "Training episodes", "return_vs_episodes.png"),
        ("minutes", "Training-loop time (minutes)", "return_vs_training_time.png"),
    ):
        fig, ax = plt.subplots(figsize=(8.5, 5.5), constrained_layout=True)
        for arm, label in (("gru", "Full GRU"), ("frozen", "Frozen after 10k"),
                           ("rssm", "RSSM encoder")):
            points = []
            for episodes in (10_000, 12_500, 15_000):
                group = [record for record in records
                         if record["arm"] == arm and int(record["episodes"]) == episodes]
                if not group:
                    continue
                y = interval([record["mean_return"] for record in group])
                x = (episodes if xkey == "episodes" else
                     float(np.mean([record["training_loop_seconds"] for record in group])) / 60.0)
                lower, upper = y["ci95_across_training_seeds"]
                points.append((x, y["mean"], y["mean"] - lower, upper - y["mean"]))
            if points:
                points.sort(key=lambda item: item[0])
                ax.errorbar(
                    [point[0] for point in points], [point[1] for point in points],
                    yerr=np.asarray([[point[2] for point in points],
                                     [point[3] for point in points]]),
                    marker="o", capsize=3, label=label,
                )
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Mean discounted return")
        ax.set_title("q observed; noisy grid position")
        ax.grid(alpha=0.25)
        ax.legend()
        fig.savefig(output_dir / filename, dpi=170)
        plt.close(fig)

    print(f"Saved convergence analysis: {json_path}", flush=True)


if __name__ == "__main__":
    main()
