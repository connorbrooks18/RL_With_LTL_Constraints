"""Run a serial, paired end-to-end versus frozen-GRU DQN comparison.

Each seed gets the same total episode budget and epsilon schedule in both arms.
The frozen arm trains GRU and Q head jointly for the first half, then freezes
the GRU and trains only the linear Q head. Runs alternate order by seed to
reduce bias from machine load or thermal changes over time.
"""

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


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_completed(path: Path) -> dict:
    if not path.exists():
        return {"schema_version": 1, "config": {}, "runs": []}
    return json.loads(path.read_text(encoding="utf-8"))


def latest_checkpoint(checkpoint_dir: Path, episodes: int, horizon: int = 600) -> Path:
    candidates = list(checkpoint_dir.glob(f"*_K{episodes}_T{horizon}_*.pt"))
    if not candidates:
        raise FileNotFoundError(
            f"No {episodes}-episode checkpoint found in {checkpoint_dir}"
        )
    return max(candidates, key=lambda path: path.stat().st_mtime)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(18, 27)))
    parser.add_argument("--episodes", type=int, default=15_000)
    parser.add_argument(
        "--epsilon-decay-fraction", type=float, default=0.25,
        help="fraction of the full episode budget over which epsilon decays",
    )
    parser.add_argument(
        "--frozen-episodes", type=int,
        help="optional shorter stop episode for the frozen-GRU arm",
    )
    parser.add_argument(
        "--frozen-epsilon-decay-fraction", type=float,
        help="override the frozen arm's fraction; by default it reaches the epsilon floor on the full arm's same absolute episode",
    )
    parser.add_argument("--freeze-after", type=int, default=10_000)
    parser.add_argument("--horizon", type=int, default=600)
    parser.add_argument(
        "--max-runs", type=int,
        help="optional serial batch limit; the manifest resumes on the next invocation",
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("artifacts/gru_freeze_ab/confirmatory_10k/training_manifest.json"),
    )
    args = parser.parse_args()
    if not 0 < args.freeze_after < args.episodes:
        parser.error("--freeze-after must be between 1 and --episodes-1")
    frozen_episodes = args.frozen_episodes or args.episodes
    if not args.freeze_after < frozen_episodes <= args.episodes:
        parser.error("--frozen-episodes must be after --freeze-after and no later than --episodes")
    frozen_epsilon_decay_fraction = (
        args.frozen_epsilon_decay_fraction
        if args.frozen_epsilon_decay_fraction is not None
        else args.epsilon_decay_fraction * args.episodes / frozen_episodes
    )
    if not 0.0 < frozen_epsilon_decay_fraction <= 1.0:
        parser.error("frozen-arm epsilon decay fraction must be in (0, 1]")
    if args.horizon != 600:
        parser.error("This registered comparison uses a 600-step horizon")

    manifest_path = args.output.resolve()
    log_dir = manifest_path.parent / "logs"
    checkpoint_root = ROOT / "checkpoints_gru_freeze_ab"
    manifest = read_completed(manifest_path)
    wanted_config = {
        "seeds": args.seeds,
        "episodes": args.episodes,
        "freeze_after_episodes": args.freeze_after,
        "horizon": args.horizon,
        "history_length": 4,
        "learning_rate": 1e-4,
        "batch_size": 32,
        "replay_capacity": 100_000,
        "target_update_every_environment_steps": 50,
        "update_every_environment_steps": 4,
        "epsilon_floor": 0.1,
        "epsilon_decay_fraction": args.epsilon_decay_fraction,
        "wall_time_method": "serial subprocess wall clock; arm order alternates by seed",
    }
    if args.frozen_episodes is not None or args.frozen_epsilon_decay_fraction is not None:
        wanted_config["frozen_episodes"] = frozen_episodes
        wanted_config["frozen_epsilon_decay_fraction"] = frozen_epsilon_decay_fraction
    if manifest.get("runs") and manifest.get("config") != wanted_config:
        raise SystemExit(
            "Existing manifest config differs. Use a new --output path for a new design."
        )
    manifest["config"] = wanted_config
    manifest.setdefault("runs", [])
    complete = {
        (int(item["training_seed"]), item["arm"])
        for item in manifest["runs"] if item.get("status") == "complete"
    }
    for item in manifest["runs"]:
        if item.get("status") != "complete" or item.get("arm") != "frozen":
            continue
        seed = int(item["training_seed"])
        checkpoint_dir = checkpoint_root / f"seed{seed}_frozen"
        milestones = []
        for checkpoint in sorted(checkpoint_dir.glob("*_headonly.pt")):
            metadata = json.loads(
                checkpoint.with_suffix(".json").read_text(encoding="utf-8")
            )
            milestones.append({
                "checkpoint": str(checkpoint.relative_to(ROOT)),
                "metadata": str(checkpoint.with_suffix(".json").relative_to(ROOT)),
                "run_id": metadata["run_id"],
                "episodes": int(metadata["episodes"]),
                "training_loop_seconds": float(metadata["training_loop_seconds"]),
            })
        if milestones:
            item["head_only_progress_checkpoints"] = milestones
    write_json(manifest_path, manifest)

    roster = []
    for seed_index, seed in enumerate(args.seeds):
        order = ("full", "frozen") if seed_index % 2 == 0 else ("frozen", "full")
        roster.extend((seed, arm) for arm in order)

    completed_this_batch = 0
    for index, (seed, arm) in enumerate(roster, start=1):
        if (seed, arm) in complete:
            print(f"[{index}/{len(roster)}] seed {seed} {arm}: already complete", flush=True)
            continue
        if args.max_runs is not None and completed_this_batch >= args.max_runs:
            break

        checkpoint_dir = checkpoint_root / f"seed{seed}_{arm}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"seed{seed}_{arm}.log"
        command = [
            sys.executable, str(ROOT / "train_pomdp.py"),
            "--seed", str(seed),
            "--episodes", str(args.episodes if arm == "full" else frozen_episodes),
            "--horizon", str(args.horizon),
            "--history-length", "4",
            "--learning-rate", "1e-4",
            "--batch-size", "32",
            "--replay-capacity", "100000",
            "--target-update-every", "50",
            "--epsilon-floor", "0.1",
            "--epsilon-decay-fraction", str(
                args.epsilon_decay_fraction
                if arm == "full" else frozen_epsilon_decay_fraction
            ),
            "--checkpoint-dir", str(checkpoint_dir),
        ]
        if arm == "frozen":
            command.extend([
                "--freeze-gru-after-episodes", str(args.freeze_after),
            ])

        started = utc_now()
        started_perf = time.perf_counter()
        print(
            f"[{index}/{len(roster)}] starting seed {seed} {arm} at {started}; "
            f"log: {log_path}",
            flush=True,
        )
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8") as log_file:
            child_env = os.environ.copy()
            child_env["PYTHONUNBUFFERED"] = "1"
            completed_process = subprocess.run(
                command,
                cwd=ROOT,
                env=child_env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                check=False,
                text=True,
            )
        elapsed = time.perf_counter() - started_perf
        finished = utc_now()

        record = {
            "training_seed": seed,
            "arm": arm,
            "episodes": args.episodes if arm == "full" else frozen_episodes,
            "status": "complete" if completed_process.returncode == 0 else "failed",
            "started_at_utc": started,
            "finished_at_utc": finished,
            "process_wall_seconds": elapsed,
            "returncode": completed_process.returncode,
            "command": command,
            "stdout_log": str(log_path.relative_to(ROOT)),
        }
        if completed_process.returncode == 0:
            arm_episodes = args.episodes if arm == "full" else frozen_episodes
            checkpoint = latest_checkpoint(checkpoint_dir, arm_episodes, args.horizon)
            metadata_path = checkpoint.with_suffix(".json")
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            record.update({
                "checkpoint": str(checkpoint.relative_to(ROOT)),
                "metadata": str(metadata_path.relative_to(ROOT)),
                "run_id": metadata["run_id"],
                "training_loop_seconds": metadata["training_loop_seconds"],
                "joint_gru_head_training_seconds": metadata[
                    "joint_gru_head_training_seconds"
                ],
                "head_only_training_seconds": metadata["head_only_training_seconds"],
            })
            if arm == "frozen":
                snapshot = latest_checkpoint(checkpoint_dir, args.freeze_after, args.horizon)
                record["pre_freeze_checkpoint"] = str(snapshot.relative_to(ROOT))
                record["pre_freeze_run_id"] = json.loads(
                    snapshot.with_suffix(".json").read_text(encoding="utf-8")
                )["run_id"]
                record["head_only_progress_checkpoints"] = [
                    {
                        "checkpoint": str(path.relative_to(ROOT)),
                        "metadata": str(path.with_suffix(".json").relative_to(ROOT)),
                        "run_id": json.loads(
                            path.with_suffix(".json").read_text(encoding="utf-8")
                        )["run_id"],
                        "episodes": int(json.loads(
                            path.with_suffix(".json").read_text(encoding="utf-8")
                        )["episodes"]),
                        "training_loop_seconds": float(json.loads(
                            path.with_suffix(".json").read_text(encoding="utf-8")
                        )["training_loop_seconds"]),
                    }
                    for path in sorted(checkpoint_dir.glob("*_headonly.pt"))
                ]
        manifest["runs"].append(record)
        write_json(manifest_path, manifest)
        completed_this_batch += 1
        print(
            f"[{index}/{len(roster)}] finished seed {seed} {arm}: "
            f"{elapsed:.1f}s, returncode={completed_process.returncode}",
            flush=True,
        )
        if completed_process.returncode != 0:
            raise SystemExit(
                f"Training failed; inspect {log_path}. Manifest is saved and resumable."
            )

    print(f"All requested runs complete. Manifest: {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
