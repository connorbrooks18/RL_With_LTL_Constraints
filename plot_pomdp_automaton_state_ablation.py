"""Render matched automaton-state ablation results with the notebook grid plotter."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from evaluate_pomdp import build_environment


ROOT = Path(__file__).resolve().parent


def plot_grid(grid_mdp, values, title, output, *, difference=False, label=None):
    grid_mdp.plot(value=values)
    figure = plt.gcf()
    axes = figure.axes[0]
    axes.set_title(title, fontsize=13)
    for artist in axes.texts:
        text = artist.get_text().strip("$")
        try:
            float(text)
        except ValueError:
            artist.set_fontsize(17)
        else:
            artist.set_fontsize(12)
    image = axes.images[0]
    image.set_cmap("RdBu_r" if difference else "viridis")
    if difference:
        finite = values[np.isfinite(values)]
        extent = max(float(np.max(np.abs(finite))), 1e-8)
        image.set_clim(-extent, extent)
    else:
        image.set_clim(0.0, 1.0)
    figure.colorbar(
        image, ax=axes, fraction=.046, pad=.04,
        label=label or ("Paired return difference" if difference else "Mean discounted return"),
    )
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=175, bbox_inches="tight")
    plt.close(figure)


def start_grid(starts, policy_key, field, shape):
    values = np.full(shape, np.nan, dtype=float)
    for start in starts:
        row, col = map(int, start["square"])
        item = start[field][policy_key]
        values[row, col] = float(item["mean"] if field == "policy_returns" else item["mean_difference"])
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evaluation", type=Path,
        default=Path("artifacts/pomdp_automaton_state_ablation/evaluation_q_observed.json"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("artifacts/pomdp_automaton_state_ablation"),
    )
    args = parser.parse_args()
    evaluation_path = args.evaluation if args.evaluation.is_absolute() else ROOT / args.evaluation
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    data = json.loads(evaluation_path.read_text(encoding="utf-8"))
    grid_mdp, _oa, _csrl, *_ = build_environment(
        float(data["evaluation"]["position_observation_error"])
    )
    shape = tuple(grid_mdp.shape)
    figure_specs = [
        ("history4_gru", "History-4 GRU", "gru_returns_by_start.png", "policy_returns"),
        ("current_observation_mlp", "Current-observation MLP", "mlp_no_q_returns_by_start.png", "policy_returns"),
        ("product_observation_mlp", "q-observed MLP, noisy position", "mlp_q_observed_returns_by_start.png", "policy_returns"),
        ("mdp_vi_noisy_input", "Value iteration, inferred q and noisy position", "vi_noisy_input_returns_by_start.png", "policy_returns"),
        ("mdp_vi_q_observed_noisy_position", "Value iteration, true q and noisy position", "vi_q_observed_returns_by_start.png", "policy_returns"),
        ("mdp_vi_full_state_oracle", "Full-state value-iteration oracle", "vi_full_state_oracle_returns_by_start.png", "policy_returns"),
        ("history4_gru_minus_product_mlp", "History-4 GRU − q-observed MLP", "gru_minus_q_mlp_by_start.png", "paired_differences"),
        ("product_mlp_minus_partial_mlp", "q-observed MLP − current-observation MLP", "q_mlp_minus_no_q_mlp_by_start.png", "paired_differences"),
        ("history4_gru_minus_q_observed_vi", "History-4 GRU − q-observed value iteration", "gru_minus_q_vi_by_start.png", "paired_differences"),
        ("product_mlp_minus_q_observed_vi", "q-observed MLP − q-observed value iteration", "q_mlp_minus_q_vi_by_start.png", "paired_differences"),
    ]
    lines = [
        "# Automaton-state ablation grid figures", "",
        "These maps use the notebook `GridMDP.plot` visualizer. Return maps show "
        "the mean discounted rollout return from each traversable start square. "
        "Difference maps show paired return differences on common rollout scenarios. "
        "Numbers are empirical rollout returns, not neural-network state values.", "",
        "| Figure | Description |", "|:--|:--|",
    ]
    for key, title, filename, field in figure_specs:
        values = start_grid(data["starts"], key, field, shape)
        difference = field == "paired_differences"
        plot_grid(grid_mdp, values, title, output_dir / filename,
                  difference=difference,
                  label=title.split(" · ")[-1] if difference else None)
        lines.append(f"| [{filename}]({filename}) | {title} |")
    for filename, title in (
        ("history4_gru_policy.png", "History-4 GRU noise-averaged action/Q map at q₀"),
        ("no_q_mlp_policy.png", "No-q MLP noise-averaged action/Q map"),
        ("q_observed_mlp_policy.png", "q-observed MLP noise-averaged action/Q map at q₀"),
    ):
        lines.append(f"| [{filename}]({filename}) | {title} |")
    lines.extend(["", "Regenerate with `venv/bin/python plot_pomdp_automaton_state_ablation.py`."])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "index.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved {len(figure_specs)} figures to {output_dir}")


if __name__ == "__main__":
    main()
