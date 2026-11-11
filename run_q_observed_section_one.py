"""Train the corrected Section One cohorts serially and save a run manifest."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parent
SEEDS = (17, 18, 19, 20, 21)
EPISODES = 15_000
HORIZON = 600
CHECKPOINT_EPISODES = (10_000, 12_500)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_manifest(path: Path) -> dict:
    if not path.exists():
        return {"schema_version": 1, "config": {}, "runs": []}
    return json.loads(path.read_text(encoding="utf-8"))


def write_manifest(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def checkpoint_for(directory: Path, episodes: int) -> Path:
    matches = list(directory.glob(f"*_K{episodes}_T{HORIZON}_*.pt"))
    if not matches:
        raise FileNotFoundError(f"Missing K={episodes} checkpoint in {directory}")
    return max(matches, key=lambda path: path.stat().st_mtime)


def roster():
    arms = ("gru", "mlp")
    for index, seed in enumerate(SEEDS):
        order = arms if index % 2 == 0 else tuple(reversed(arms))
        for arm in order:
            yield seed, arm
    for index, seed in enumerate(SEEDS):
        order = ("frozen", "rssm") if index % 2 == 0 else ("rssm", "frozen")
        for arm in order:
            yield seed, arm


def command_for(seed: int, arm: str, checkpoint_dir: Path) -> list[str]:
    architecture = "gru" if arm == "frozen" else arm
    command = [
        sys.executable, str(ROOT / "train_pomdp.py"),
        "--seed", str(seed),
        "--episodes", str(EPISODES),
        "--horizon", str(HORIZON),
        "--history-length", "4",
        "--architecture", architecture,
        "--reveal-automaton-state",
        "--learning-rate", "1e-4",
        "--batch-size", "32",
        "--replay-update-every", "4",
        "--replay-capacity", "100000",
        "--target-update-every", "50",
        "--epsilon-floor", "0.1",
        "--epsilon-decay-fraction", "0.25",
        "--checkpoint-dir", str(checkpoint_dir),
    ]
    if arm in ("gru", "rssm"):
        command.extend([
            "--checkpoint-episodes",
            *(str(episode) for episode in CHECKPOINT_EPISODES),
        ])
    if arm == "frozen":
        command.extend([
            "--freeze-gru-after-episodes", "10000",
            "--checkpoint-episodes", "12500",
        ])
    if arm == "rssm":
        command.extend([
            "--rssm-hidden-size", "16",
            "--rssm-stochastic-variables", "2",
            "--rssm-categorical-classes", "4",
        ])
    return command


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path,
        default=Path("artifacts/q_observed_section_one/training_manifest.json"),
    )
    parser.add_argument(
        "--max-runs", type=int,
        help="optional serial batch limit; rerunning resumes from the manifest",
    )
    parser.add_argument(
        "--arms", nargs="+", choices=("gru", "mlp", "frozen", "rssm"),
        default=("gru", "mlp", "frozen", "rssm"),
        help="cohorts to train; full GRU is the shared reference for both sets",
    )
    args = parser.parse_args()
    if args.max_runs is not None and args.max_runs < 1:
        parser.error("--max-runs must be positive")

    manifest_path = args.output.resolve()
    artifact_root = manifest_path.parent
    checkpoint_root = ROOT / "checkpoints_q_observed_section_one"
    config = {
        "seeds": list(SEEDS),
        "episodes": EPISODES,
        "steps_per_episode": HORIZON,
        "history_length": 4,
        "automaton_state_observed": True,
        "grid_position_partially_observed": True,
        "persistent_epsilon_target_memory": False,
        "observation_error": 0.25,
        "learning_rate": 1e-4,
        "batch_size": 32,
        "replay_capacity": 100_000,
        "replay_update_every_environment_steps": 4,
        "target_update_every_environment_steps": 50,
        "epsilon_floor": 0.1,
        "epsilon_decay_fraction": 0.25,
        "automaton_action_probability": 0.2,
        "freeze_after_episodes": 10_000,
        "rssm_deterministic_size": 16,
        "rssm_stochastic_variables": 2,
        "rssm_categorical_classes": 4,
        "execution": "serial; paired training seeds; alternating within-cohort order",
        "arms": list(args.arms),
    }
    manifest = read_manifest(manifest_path)
    previous_config = manifest.get("config", {})
    if manifest.get("runs") and previous_config:
        comparable_previous = dict(previous_config)
        comparable_previous.pop("arms", None)
        comparable_current = dict(config)
        comparable_current.pop("arms", None)
        if comparable_previous != comparable_current:
            raise SystemExit(
                "Existing manifest uses a different experiment design; "
                "choose a new --output path."
            )
        config["arms"] = sorted(set(previous_config.get("arms", ())) | set(args.arms))
    manifest["config"] = config
    manifest.setdefault("runs", [])

    complete = {
        (int(run["training_seed"]), run["arm"])
        for run in manifest["runs"] if run.get("status") == "complete"
    }
    selected_arms = set(args.arms)
    tasks = [(seed, arm) for seed, arm in roster() if arm in selected_arms]
    finished_this_batch = 0
    for index, (seed, arm) in enumerate(tasks, start=1):
        if (seed, arm) in complete:
            print(f"[{index}/{len(tasks)}] seed {seed} {arm}: already complete", flush=True)
            continue
        if args.max_runs is not None and finished_this_batch >= args.max_runs:
            break

        checkpoint_dir = checkpoint_root / f"seed{seed}_{arm}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        log_path = artifact_root / "logs" / f"seed{seed}_{arm}.log"
        command = command_for(seed, arm, checkpoint_dir)
        started_at = utc_now()
        started_perf = time.perf_counter()
        print(
            f"[{index}/{len(tasks)}] starting seed {seed} {arm} at {started_at}; "
            f"log: {log_path}", flush=True,
        )
        log_path.parent.mkdir(parents=True, exist_ok=True)
        child_env = os.environ.copy()
        child_env["PYTHONUNBUFFERED"] = "1"
        with log_path.open("w", encoding="utf-8") as log_file:
            result = subprocess.run(
                command, cwd=ROOT, env=child_env, stdout=log_file,
                stderr=subprocess.STDOUT, check=False, text=True,
            )
        process_seconds = time.perf_counter() - started_perf

        record = {
            "training_seed": seed,
            "arm": arm,
            "episodes": EPISODES,
            "steps_per_episode": HORIZON,
            "status": "failed" if result.returncode else "complete",
            "started_at_utc": started_at,
            "finished_at_utc": utc_now(),
            "process_wall_seconds": process_seconds,
            "returncode": result.returncode,
            "command": command,
            "stdout_log": str(log_path.relative_to(ROOT)),
            "checkpoint_dir": str(checkpoint_dir.relative_to(ROOT)),
        }
        if result.returncode == 0:
            final_checkpoint = checkpoint_for(checkpoint_dir, EPISODES)
            metadata_path = final_checkpoint.with_suffix(".json")
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if not metadata.get("partial_observation") or not metadata.get(
                "automaton_state_observed"
            ):
                record["status"] = "failed"
                record["validation_error"] = (
                    "checkpoint metadata does not confirm q-observed, "
                    "partially observed grid inputs"
                )
            else:
                record.update({
                    "checkpoint": str(final_checkpoint.relative_to(ROOT)),
                    "metadata": str(metadata_path.relative_to(ROOT)),
                    "run_id": metadata["run_id"],
                    "training_loop_seconds": float(metadata["training_loop_seconds"]),
                    "training_return_summary": metadata["training_return_summary"],
                })
                if arm != "mlp":
                    for episode in CHECKPOINT_EPISODES:
                        progress = checkpoint_for(checkpoint_dir, episode)
                        progress_metadata = json.loads(
                            progress.with_suffix(".json").read_text(encoding="utf-8")
                        )
                        record.setdefault("progress_checkpoints", []).append({
                            "episodes": episode,
                            "checkpoint": str(progress.relative_to(ROOT)),
                            "training_loop_seconds": float(
                                progress_metadata["training_loop_seconds"]
                            ),
                        })
        manifest["runs"].append(record)
        write_manifest(manifest_path, manifest)
        print(
            f"[{index}/{len(tasks)}] finished seed {seed} {arm}: "
            f"{record['status']}, process wall {process_seconds:.1f}s",
            flush=True,
        )
        finished_this_batch += 1
        if result.returncode != 0:
            print(f"Inspect {log_path} before resuming.", flush=True)
            break

    write_manifest(manifest_path, manifest)
    print(f"Manifest: {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
