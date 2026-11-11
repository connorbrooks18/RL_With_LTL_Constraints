"""Create consistent notebook-style Q/action maps for every trained checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from plot_pomdp_policy_checkpoint import plot_checkpoint


ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path,
        default=Path("artifacts/q_observed_section_one/training_manifest.json"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("artifacts/q_observed_section_one/policy_maps"),
    )
    args = parser.parse_args()
    torch.set_num_threads(1)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    figures = []
    for run in manifest.get("runs", []):
        if run.get("status") != "complete":
            continue
        candidates = [{
            "episodes": int(run["episodes"]),
            "checkpoint": run["checkpoint"],
        }]
        if run["arm"] != "mlp":
            candidates.extend(run.get("progress_checkpoints", []))
        for item in candidates:
            checkpoint = Path(item["checkpoint"])
            if not checkpoint.is_absolute():
                checkpoint = ROOT / checkpoint
            episodes = int(item["episodes"])
            output = args.output_dir / (
                f"{run['arm']}_seed{run['training_seed']}_K{episodes}.png"
            )
            if output.exists():
                print(f"Already plotted {output}", flush=True)
            else:
                plot_checkpoint(checkpoint, output)
                print(f"Saved {output}", flush=True)
            figures.append({
                "arm": run["arm"],
                "seed": int(run["training_seed"]),
                "episodes": episodes,
                "path": output.name,
            })
    lines = [
        "# q-observed policy maps", "",
        "Notebook-style policy visualizations use the existing POMDP Q/action "
        "map renderer. Each map shows the q0 controller response under the "
        "same 0.25 noisy grid-position channel; colors show expected max Q "
        "and arrows show the most common greedy action across sensor readings.",
        "",
    ]
    for arm in ("gru", "mlp", "frozen", "rssm"):
        subset = [figure for figure in figures if figure["arm"] == arm]
        if not subset:
            continue
        lines.extend([f"## {arm.upper()}", ""])
        lines.append("| Seed | Episodes | Map |")
        lines.append("|---:|---:|:--|")
        for figure in sorted(subset, key=lambda value: (value["seed"], value["episodes"])):
            lines.append(
                f"| {figure['seed']} | {figure['episodes']:,} | "
                f"[policy map]({figure['path']}) |"
            )
        lines.append("")
    (args.output_dir / "index.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved {args.output_dir / 'index.md'}", flush=True)


if __name__ == "__main__":
    main()
