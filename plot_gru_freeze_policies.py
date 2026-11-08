"""Create a policy-map image for each trained and intermediate frozen-GRU checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from plot_pomdp_policy_checkpoint import plot_checkpoint


ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pair", type=int, default=0)
    args = parser.parse_args()

    manifest_path = args.manifest.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    freeze_episode = int(manifest["config"]["freeze_after_episodes"])
    specs = []
    for run in manifest["runs"]:
        if run.get("status") != "complete":
            continue
        seed = int(run["training_seed"])
        arm = run["arm"]
        if arm == "full":
            specs.append((seed, "full", int(manifest["config"]["episodes"]), run["checkpoint"]))
            continue

        specs.append((seed, "frozen", freeze_episode, run["pre_freeze_checkpoint"]))
        for milestone in run.get("head_only_progress_checkpoints", []):
            specs.append((seed, "frozen", int(milestone["episodes"]), milestone["checkpoint"]))
        # A frozen arm can stop before the full-training episode budget. Use
        # its recorded endpoint so map labels match the checkpoint metadata.
        specs.append((seed, "frozen", int(run.get("episodes", manifest["config"]["episodes"])), run["checkpoint"]))

    rows = []
    for seed, arm, episodes, checkpoint in sorted(specs):
        label = f"seed{seed}_{arm}_{episodes}episodes"
        output = output_dir / f"{label}.png"
        metadata = plot_checkpoint(ROOT / checkpoint, output, args.pair)
        rows.append((seed, arm, episodes, output.name, metadata["run_id"]))
        print(f"Saved {output}")

    index_path = output_dir / "index.md"
    lines = [
        "# Frozen-GRU policy maps",
        "",
        "Each map shows the noise-averaged greedy action and expected maximum Q "
        "value over the task grid. For GRU policies, panels show a one-observation "
        "cold start and a four-step repeated-observation probe.",
        "",
        "| Training seed | Arm | Episodes | Run ID | Policy map |",
        "|---:|:---|---:|:---|:---|",
    ]
    for seed, arm, episodes, filename, run_id in rows:
        lines.append(
            f"| {seed} | {arm} | {episodes:,} | `{run_id}` | "
            f"[view]({filename}) |"
        )
    index_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved {index_path}")


if __name__ == "__main__":
    main()
