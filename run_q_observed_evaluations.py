"""Run all q-observed evaluations and visualizations serially from the manifest."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parent


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path,
        default=Path("artifacts/q_observed_section_one/training_manifest.json"),
    )
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--horizon", type=int, default=1_000)
    parser.add_argument("--convergence-runs", type=int, default=20)
    parser.add_argument("--convergence-horizon", type=int, default=500)
    args = parser.parse_args()
    manifest_path = args.manifest.resolve()
    manifest = read_json(manifest_path)
    seeds = list(map(int, manifest["config"]["seeds"]))
    by_arm_seed = {}
    for run in manifest.get("runs", []):
        if run.get("status") == "complete":
            by_arm_seed[(run["arm"], int(run["training_seed"]))] = run
    for arm in ("gru", "mlp", "frozen", "rssm"):
        missing = [seed for seed in seeds if (arm, seed) not in by_arm_seed]
        if missing:
            raise SystemExit(f"Training is incomplete for {arm}; missing seeds {missing}")

    root = manifest_path.parent
    output = root / "evaluation_manifest.json"
    state = read_json(output) if output.exists() else {
        "schema_version": 1, "config": {}, "runs": []
    }
    config = {
        "training_seeds": seeds,
        "final_rollouts_per_start": args.runs,
        "final_horizon": args.horizon,
        "convergence_rollouts_per_start": args.convergence_runs,
        "convergence_horizon": args.convergence_horizon,
        "serial": True,
        "q_observed_grid_partially_observed": True,
    }
    if state.get("runs") and state.get("config") != config:
        raise SystemExit("Existing evaluation manifest config differs; use a new artifact directory")
    state["config"] = config
    state.setdefault("runs", [])
    completed = {item["name"] for item in state["runs"] if item.get("status") == "complete"}

    gru = [str((ROOT / by_arm_seed[("gru", seed)]["checkpoint"]).resolve()) for seed in seeds]
    mlp = [str((ROOT / by_arm_seed[("mlp", seed)]["checkpoint"]).resolve()) for seed in seeds]
    jobs = [
        ("history4_gru_vs_mlp", [
            sys.executable, str(ROOT / "evaluate_pomdp_gru_vs_no_gru.py"),
            "--gru", *gru, "--no-gru", *mlp,
            "--runs", str(args.runs), "--horizon", str(args.horizon),
            "--seed", "17", "--workers", "1",
            "--output", str(root / "history4_gru_vs_mlp.json"),
        ]),
        ("history4_gru_vs_mlp_grid_visuals", [
            sys.executable, str(ROOT / "plot_q_observed_history4_comparison.py"),
            str(root / "history4_gru_vs_mlp.json"),
            "--output-dir", str(root / "history4_comparison"),
        ]),
        ("convergence", [
            sys.executable, str(ROOT / "evaluate_q_observed_convergence.py"),
            "--manifest", str(manifest_path),
            "--runs", str(args.convergence_runs),
            "--horizon", str(args.convergence_horizon),
            "--episodes", "10000", "12500", "15000",
            "--seed", "20261110",
            "--output-dir", str(root / "convergence"),
        ]),
        ("final_quality_with_noisy_vi", [
            sys.executable, str(ROOT / "evaluate_q_observed_convergence.py"),
            "--manifest", str(manifest_path),
            "--runs", str(args.runs), "--horizon", str(args.horizon),
            "--episodes", "15000", "--include-vi",
            "--seed", "17",
            "--output-dir", str(root / "final_quality"),
        ]),
        ("final_grid_values", [
            sys.executable, str(ROOT / "plot_q_observed_final_grids.py"),
            "--comparison", str(root / "history4_gru_vs_mlp.json"),
            "--final-quality", str(root / "final_quality" / "evaluation.json"),
            "--output-dir", str(root / "final_quality"),
        ]),
        ("policy_gallery", [
            sys.executable, str(ROOT / "generate_q_observed_policy_gallery.py"),
            "--manifest", str(manifest_path),
            "--output-dir", str(root / "policy_maps"),
        ]),
    ]
    logs = root / "evaluation_logs"
    logs.mkdir(parents=True, exist_ok=True)
    for index, (name, command) in enumerate(jobs, 1):
        if name in completed:
            print(f"[{index}/{len(jobs)}] {name}: already complete", flush=True)
            continue
        log_path = logs / f"{name}.log"
        started = utc_now()
        t0 = time.perf_counter()
        print(f"[{index}/{len(jobs)}] starting {name}; log: {log_path}", flush=True)
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        with log_path.open("w", encoding="utf-8") as log:
            result = subprocess.run(
                command, cwd=ROOT, env=env, stdout=log,
                stderr=subprocess.STDOUT, check=False, text=True,
            )
        elapsed = time.perf_counter() - t0
        record = {
            "name": name,
            "status": "complete" if result.returncode == 0 else "failed",
            "started_at_utc": started,
            "finished_at_utc": utc_now(),
            "process_wall_seconds": elapsed,
            "returncode": result.returncode,
            "command": command,
            "log": str(log_path.relative_to(ROOT)),
        }
        state["runs"].append(record)
        write_json(output, state)
        print(f"[{index}/{len(jobs)}] finished {name}: {record['status']} ({elapsed:.1f}s)", flush=True)
        if result.returncode:
            raise SystemExit(f"{name} failed; inspect {log_path}")
    print(f"Evaluation manifest: {output}", flush=True)


if __name__ == "__main__":
    main()
