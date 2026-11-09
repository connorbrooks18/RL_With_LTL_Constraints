"""Render evaluation-return grids with the same GridMDP visualizer as the notebooks."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from evaluate_pomdp import build_environment


ROOT = Path(__file__).resolve().parent


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def add_run_grid(accumulator, run: dict, metric: str) -> None:
    for start in run["starts"]:
        row, col = map(int, start["square"])
        if start.get(metric) is None:
            continue
        value = start[metric]
        if isinstance(value, dict):
            value = value["mean"]
        accumulator[row, col].append(float(value))


def mean_grid(runs: list[dict], metric: str, shape: tuple[int, int]) -> np.ndarray:
    samples = defaultdict(list)
    for run in runs:
        add_run_grid(samples, run, metric)
    values = np.full(shape, np.nan, dtype=float)
    for (row, col), entries in samples.items():
        values[row, col] = float(np.mean(entries))
    return values


def paired_difference_grid(
    runs: list[dict], metric: str, shape: tuple[int, int]
) -> np.ndarray:
    return mean_grid(runs, metric, shape)


def plot_grid(
    grid_mdp,
    values: np.ndarray,
    title: str,
    output: Path,
    *,
    cmap: str = "viridis",
    limits: tuple[float, float] | None = None,
    colorbar_label: str = "Mean discounted return",
) -> None:
    grid_mdp.plot(value=values)
    figure = plt.gcf()
    axes = figure.axes[0]
    axes.set_title(title, fontsize=14)
    for artist in axes.texts:
        text = artist.get_text().strip("$")
        try:
            float(text)
        except ValueError:
            artist.set_fontsize(18)
        else:
            artist.set_fontsize(13)
    image = axes.images[0]
    image.set_cmap(cmap)
    if limits is None:
        finite = values[np.isfinite(values)]
        if cmap in ("RdBu", "RdBu_r", "coolwarm"):
            extent = max(float(np.max(np.abs(finite))), 1e-8)
            limits = (-extent, extent)
        else:
            limits = (float(np.min(finite)), float(np.max(finite)))
    image.set_clim(*limits)
    figure.colorbar(image, ax=axes, fraction=.046, pad=.04, label=colorbar_label)
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=175, bbox_inches="tight")
    plt.close(figure)


def write_index(rows: list[tuple[str, str, str]], output_dir: Path) -> None:
    lines = [
        "# Notebook-style state-grid visualizations",
        "",
        "Figures use `GridMDP.plot(value=...)`, the same grid renderer used in the "
        "POMDP notebooks. Cell numbers are empirical mean discounted rollout "
        "returns by start square, or paired return differences where marked. "
        "They are not learned scalar state-value estimates.",
        "",
        "| Group | Figure | Description |",
        "|:--|:--|:--|",
    ]
    for group, filename, description in rows:
        lines.append(f"| {group} | [{filename}]({filename}) | {description} |")
    lines.extend([
        "",
        "Regenerate these figures with `venv/bin/python plot_research_summary_grids.py`.",
    ])
    (output_dir / "index.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def checkpoint_episodes(run: dict) -> int:
    path = Path(run["checkpoint"])
    try:
        import re

        match = re.search(r"_K(\d+)_", path.name)
        if match:
            return int(match.group(1))
    except (KeyError, TypeError):
        pass
    raise ValueError(f"Cannot infer checkpoint episode count: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("artifacts/research_summary/grid_visualizations"),
    )
    args = parser.parse_args()
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    grid_mdp, _oa, _csrl, *_ = build_environment(0.25)
    shape = tuple(grid_mdp.shape)
    rows: list[tuple[str, str, str]] = []

    # Ten-seed experiment: paired GRU and no-history MLP versus notebook MDP baselines.
    architecture = load_json(ROOT / "artifacts/pomdp_history4_vs_feedforward_dqn_10seeds.json")
    vi = load_json(ROOT / "artifacts/pomdp_history4_vs_mdp_value_iteration_10seeds.json")
    main_specs = [
        ("GRU-DQN, 15k · 10 seeds", architecture["runs"], "gru_dqn",
         "gru_15k_10seeds.png"),
        ("No-history MLP-DQN, 15k · 10 seeds", architecture["runs"], "no_gru_dqn",
         "mlp_15k_10seeds.png"),
        ("Noisy-input MDP value iteration · 10 matched seeds", vi["runs"],
         "mdp_value_iteration_on_noisy_observation", "noisy_vi_10seeds.png"),
        ("Full-state MDP VI reference · 10 matched seeds", vi["runs"],
         "mdp_value_iteration_full_state_oracle", "full_state_vi_10seeds.png"),
    ]
    for title, runs, key, filename in main_specs:
        values = mean_grid(runs, key, shape)
        path = output_dir / filename
        plot_grid(
            grid_mdp, values, title, path, limits=(0.0, 1.0),
            colorbar_label="Mean discounted return",
        )
        rows.append(("Experiment 1", filename, title))

    for title, runs, key, filename, label in (
        ("History-4 GRU − no-history MLP · paired 10-seed mean", architecture["runs"],
         "paired_difference_gru_minus_no_gru", "gru_minus_mlp_by_start.png",
         "GRU − MLP return"),
        ("History-4 GRU − noisy-input MDP VI · paired 10-seed mean", vi["runs"],
         "paired_difference_gru_minus_vi", "gru_minus_noisy_vi_by_start.png",
         "GRU − noisy-input VI"),
    ):
        values = paired_difference_grid(runs, key, shape)
        path = output_dir / filename
        finite = values[np.isfinite(values)]
        extent = max(float(np.max(np.abs(finite))), 1e-8)
        plot_grid(
            grid_mdp, values, title, path, cmap="RdBu_r",
            limits=(-extent, extent), colorbar_label=label,
        )
        rows.append(("Experiment 1", filename, title))

    # Confirmatory 10k freeze experiment: common-seed return grids and paired difference.
    freeze = load_json(ROOT / "artifacts/gru_freeze_ab/confirmatory_10k/evaluation.json")
    freeze_groups: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for run in freeze["runs"]:
        arm = "frozen" if Path(run["checkpoint"]).parent.name.endswith("_frozen") else "full"
        episodes = checkpoint_episodes(run)
        freeze_groups[(arm, episodes)].append(run)
    for (arm, episodes), runs in sorted(freeze_groups.items()):
        if episodes not in (10_000, 12_500, 15_000):
            continue
        values = mean_grid(runs, "gru_dqn", shape)
        filename = f"freeze_{arm}_{episodes}_9seeds_grid.png"
        title = f"{arm.title()} GRU-DQN, {episodes:,} episodes · confirmatory seeds"
        plot_grid(grid_mdp, values, title, output_dir / filename,
                  limits=(0.0, 1.0))
        rows.append(("GRU freeze", filename, title))

    by_seed: dict[int, dict[str, dict[tuple[int, int], float]]] = defaultdict(dict)
    for run in freeze["runs"]:
        arm = "frozen" if Path(run["checkpoint"]).parent.name.endswith("_frozen") else "full"
        episodes = checkpoint_episodes(run)
        if episodes != 15_000:
            continue
        values = {}
        for start in run["starts"]:
            square = tuple(map(int, start["square"]))
            values[square] = float(start["gru_dqn"]["mean"])
        by_seed[int(run["training_seed"])][arm] = values
    diffs = defaultdict(list)
    for arms in by_seed.values():
        if "full" not in arms or "frozen" not in arms:
            continue
        for square, value in arms["frozen"].items():
            diffs[square].append(value - arms["full"][square])
    diff_grid = np.full(shape, np.nan, dtype=float)
    for (row, col), values in diffs.items():
        diff_grid[row, col] = float(np.mean(values))
    extent = max(float(np.nanmax(np.abs(diff_grid))), 1e-8)
    filename = "freeze_frozen_minus_full_15k_by_start.png"
    title = "Frozen 15k − full GRU 15k · paired mean across 9 seeds"
    plot_grid(grid_mdp, diff_grid, title, output_dir / filename, cmap="RdBu_r",
              limits=(-extent, extent), colorbar_label="Frozen − full return")
    rows.append(("GRU freeze", filename, title))

    # Seed-27 frozen-head trajectory and full-GRU extension.
    extension = load_json(ROOT / "artifacts/gru_freeze_ab/convergence_25k/evaluation.json")
    for run in extension["runs"]:
        path = Path(run["checkpoint"])
        episodes = checkpoint_episodes(run)
        arm = "frozen" if path.parent.name.endswith("_frozen") else "full"
        values = mean_grid([run], "gru_dqn", shape)
        filename = f"seed27_{arm}_{episodes}_return_grid.png"
        title = f"Seed 27 {arm} GRU-DQN, {episodes:,} episodes"
        plot_grid(grid_mdp, values, title, output_dir / filename,
                  limits=(0.0, 1.0))
        rows.append(("Seed-27 freeze trajectory", filename, title))

    # RSSM-DQN duration sweep and matched GRU checkpoints.
    rssm = load_json(ROOT / "artifacts/rssm_dqn_pilot/evaluation.json")
    for run in rssm["runs"]:
        episodes = int(run["training_episodes"])
        arch = run["architecture"]
        frozen = bool(run.get("gru_frozen_at_checkpoint"))
        arm = "frozen" if frozen else "joint"
        values = mean_grid([run], "policy_dqn", shape)
        filename = f"seed27_{arch}_{arm}_{episodes}_return_grid.png"
        title = f"Seed 27 {arch.upper()}-{arm}, {episodes:,} episodes"
        plot_grid(grid_mdp, values, title, output_dir / filename,
                  limits=(0.0, 1.0))
        rows.append(("RSSM and GRU comparison", filename, title))

    write_index(rows, output_dir)
    print(f"Saved {len(rows)} notebook-style grid figures to {output_dir}")


if __name__ == "__main__":
    main()
