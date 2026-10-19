"""Wait for a POMDP training process, then compare its checkpoint to baseline."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import torch


def process_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def checkpoint_from_log(log_path: Path) -> Path | None:
    if not log_path.exists():
        return None
    for line in log_path.read_text(errors="replace").splitlines():
        marker = "Saved DQN checkpoint: "
        if marker in line:
            return Path(line.split(marker, 1)[1].strip())
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--training-log", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--expected-history", type=int, default=16)
    parser.add_argument("--expected-seed", type=int, default=17)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--horizon", type=int, default=1_000)
    parser.add_argument("--evaluation-seed", type=int, default=17)
    args = parser.parse_args()

    training_log = args.training_log.expanduser().resolve()
    while process_exists(args.pid):
        if checkpoint_from_log(training_log) is not None:
            break
        time.sleep(30)

    checkpoint = checkpoint_from_log(training_log)
    if checkpoint is None or not checkpoint.is_file():
        raise SystemExit(
            f"Training PID {args.pid} ended without a saved checkpoint; "
            f"inspect {training_log}"
        )
    if not checkpoint.is_absolute():
        checkpoint = (Path.cwd() / checkpoint).resolve()
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    metadata = saved["metadata"]
    if metadata.get("history_length") != args.expected_history:
        raise SystemExit(
            f"Checkpoint history_length={metadata.get('history_length')}; "
            f"expected {args.expected_history}"
        )
    if metadata.get("seed") != args.expected_seed:
        raise SystemExit(
            f"Checkpoint seed={metadata.get('seed')}; "
            f"expected {args.expected_seed}"
        )

    project_dir = Path(__file__).resolve().parent
    report_path = project_dir / "checkpoints_pomdp" / "history8_vs_history16_eval.json"
    report_log = project_dir / "checkpoints_pomdp" / "history8_vs_history16_eval.log"
    command = [
        sys.executable,
        str(project_dir / "evaluate_pomdp.py"),
        str(args.baseline.expanduser().resolve()),
        str(checkpoint),
        "--runs", str(args.runs),
        "--horizon", str(args.horizon),
        "--seed", str(args.evaluation_seed),
        "--output", str(report_path),
    ]
    with report_log.open("w", encoding="utf-8") as output:
        output.write(f"Command: {command!r}\n")
        output.flush()
        subprocess.run(
            command,
            cwd=project_dir,
            stdout=output,
            stderr=subprocess.STDOUT,
            check=True,
        )
    print(f"Matched evaluation saved to {report_path}", flush=True)
    print(f"Evaluation log: {report_log}", flush=True)


if __name__ == "__main__":
    main()
