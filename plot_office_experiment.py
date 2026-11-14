"""Notebook-style office training, outcome, and policy visualizations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import torch

from dqn import DQN, GRUDQN
from evaluate_pomdp import action_name
from office_pomdp import SIGNATURE_GROUP_BY_STATE, read_state_probabilities
from train_office_pomdp import build_environment


def _load(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    metadata = checkpoint["metadata"]
    if metadata["architecture"] == "gru":
        model = GRUDQN(
            int(metadata["n_observations"]), int(metadata["n_actions"]),
            hidden_size=int(metadata["gru_hidden_size"]),
            num_layers=int(metadata["gru_num_layers"]),
        )
    else:
        model = DQN(
            int(metadata["n_observations"]), int(metadata["n_actions"]),
            hidden_size=int(metadata["hidden_size"]),
        )
    model.load_state_dict(checkpoint["policy_state_dict"])
    model.eval()
    return model, metadata, checkpoint


def _rolling_mean(values, window):
    window = min(int(window), len(values))
    if window <= 1:
        return np.asarray(values, dtype=float)
    return np.convolve(values, np.ones(window) / window, mode="valid")


def plot_training(checkpoints, output, window):
    colors = {"gru": "#2474b5", "mlp": "#e68632"}
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    for path in checkpoints:
        _model, meta, checkpoint = _load(path)
        returns = checkpoint["episode_returns"].detach().cpu().numpy().astype(float)
        means = _rolling_mean(returns, window)
        x = np.arange(len(means)) + min(window, len(returns))
        ax.plot(x, means, color=colors[meta["architecture"]], alpha=.66,
                linewidth=1.15, label=meta["architecture"].upper())
    ax.set_xlabel("Training episode")
    ax.set_ylabel(f"Discounted training return ({window}-episode moving average)")
    ax.set_title("Junchao office: training return by architecture")
    ax.grid(alpha=.22)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_evaluation(report_path, output):
    report = json.loads(report_path.read_text(encoding="utf-8"))
    episodes = []
    return_lines = {"gru": [], "mlp": []}
    thesis_safe_lines = {"gru": [], "mlp": []}
    for episode_text, group in report["by_training_episodes"].items():
        episodes.append(int(episode_text))
        runs = group["runs"]
        for architecture in ("gru", "mlp"):
            return_values = [run["metrics"]["mean_discounted_return"][architecture]["mean"]
                             for run in runs]
            safe_values = [
                run["thesis_start_metrics"]["safe_two_cycle_success_rate"][architecture]["mean"]
                for run in runs
            ]
            for values, store in ((return_values, return_lines), (safe_values, thesis_safe_lines)):
                array = np.asarray(values, dtype=float)
                mean = float(array.mean())
                half = (4.303 * float(array.std(ddof=1)) / np.sqrt(len(array))
                        if len(array) == 3 else
                        2.776 * float(array.std(ddof=1)) / np.sqrt(len(array))
                        if len(array) == 5 else
                        1.96 * float(array.std(ddof=1)) / np.sqrt(len(array))
                        if len(array) > 1 else 0.0)
                store[architecture].append((mean, half))
    order = np.argsort(episodes)
    episodes = np.asarray(episodes)[order]
    colors = {"gru": "#2474b5", "mlp": "#e68632"}
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
    for architecture in ("gru", "mlp"):
        for ax, store, title, ylabel in (
            (axes[0], return_lines, "Mean return from all 16 rooms", "Mean discounted return"),
            (axes[1], thesis_safe_lines, "Two delivery cycles from thesis start (3,3)",
             "P(2 cycles, no storage violation)"),
        ):
            values = np.asarray(store[architecture], dtype=float)[order]
            ax.errorbar(episodes / 1000, values[:, 0], yerr=values[:, 1],
                        marker="o", capsize=3, linewidth=1.7,
                        color=colors[architecture], label=architecture.upper())
            ax.set_title(title)
            ax.set_xlabel("Training episodes (thousands)")
            ax.set_ylabel(ylabel)
            ax.grid(alpha=.22)
    axes[0].legend()
    axes[1].legend()
    fig.suptitle("Office Task 1 policy evaluation by training budget")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _policy_panel(model, meta, mdp, oa, csrl):
    q_map = np.full(mdp.shape, np.nan, dtype=float)
    action_modes = {}
    allowed = csrl.actions_for_observed_q(oa.q0)
    for row in range(4):
        for col in range(4):
            values = {}
            expected_max_q = 0.0
            for observed, probability in read_state_probabilities((row, col)).items():
                observation = (0, oa.q0, int(observed[0]), int(observed[1]))
                if meta["architecture"] == "gru":
                    encoded = csrl.encode_observation_history(
                        [observation], history_length=int(meta["history_length"]),
                        previous_actions=[None], partial_observation=True,
                        automaton_state_observed=True,
                        epsilon_commitment_mode="none",
                    ).unsqueeze(0)
                else:
                    encoded = csrl.encode_observation(
                        observation, partial_observation=True,
                        automaton_state_observed=True,
                    ).unsqueeze(0)
                with torch.no_grad():
                    q_values = model(encoded).squeeze(0).cpu().numpy()
                action = max(allowed, key=lambda a: q_values[a])
                values[action] = values.get(action, 0.0) + probability
                expected_max_q += probability * float(q_values[action])
            action = max(values, key=values.get)
            q_map[row, col] = expected_max_q
            action_modes[(row, col)] = (action, values[action])
    return q_map, action_modes


def plot_policies(gru_path, mlp_path, output):
    gru, gm, _ = _load(gru_path)
    mlp, mm, _ = _load(mlp_path)
    _mdp, _oa, csrl = build_environment(
        float(gm["discount_nonaccepting"]), float(gm["discount_accepting"])
    )
    mdp, oa = csrl.mdp, csrl.oa
    panels = [_policy_panel(gru, gm, mdp, oa, csrl), _policy_panel(mlp, mm, mdp, oa, csrl)]
    low = min(float(np.nanmin(values)) for values, _ in panels)
    high = max(float(np.nanmax(values)) for values, _ in panels)
    if high <= low:
        high = low + 1e-6
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 5.5))
    arrows = {"U": "↑", "D": "↓", "R": "→", "L": "←"}
    label_map = {"a": "a", "c": "c", "p": "p", "s": "S"}
    for ax, (values, actions), meta, title in zip(
        axes, panels, (gm, mm), ("History-4 GRU-DQN", "Current-reading MLP-DQN")
    ):
        image = ax.imshow(values, cmap="viridis", vmin=low, vmax=high, interpolation="nearest")
        ax.set_xticks(range(4))
        ax.set_yticks(range(4))
        ax.set_xticks(np.arange(-.5, 4, 1), minor=True)
        ax.set_yticks(np.arange(-.5, 4, 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.1, alpha=.8)
        ax.tick_params(which="minor", bottom=False, left=False)
        ax.xaxis.tick_top()
        ax.set_title(f"{title} · {meta['episodes']:,} episodes")
        for row in range(4):
            for col in range(4):
                props = tuple(mdp.label[row, col])
                if props:
                    ax.text(col, row - .28, label_map.get(props[0], props[0]),
                            ha="center", va="center", fontsize=8, weight="bold")
                action, probability = actions[(row, col)]
                name = action_name(action, len(mdp.A))
                ax.text(col, row + .05, arrows.get(name, f"ε{action - len(mdp.A)}"),
                        ha="center", va="center", fontsize=13, weight="bold",
                        bbox={"facecolor": "white", "edgecolor": "none", "alpha": .78, "pad": 1})
                alias_count = len(SIGNATURE_GROUP_BY_STATE[(row, col)])
                alias = "unique" if alias_count == 1 else f"×{alias_count} alias"
                ax.text(col, row + .34, f"{alias} · {probability:.1f}",
                        ha="center", va="center", fontsize=5.7, color="#222222")
        ax.scatter([3], [3], marker="o", s=100, facecolors="none",
                   edgecolors="#00a6a6", linewidths=2, zorder=5)
        fig.colorbar(image, ax=ax, fraction=.046, pad=.04, label="Expected max Q")
    fig.suptitle(
        "Office policy maps at q₀ · initial one-reading history averaged over analytic aliases\n"
        "Arrow is the modal action over possible room readings; ring marks the training start",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, .9))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-checkpoints", type=Path, nargs="+")
    parser.add_argument("--evaluation", type=Path)
    parser.add_argument("--gru-final", type=Path)
    parser.add_argument("--mlp-final", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--window", type=int, default=500)
    args = parser.parse_args()
    if args.training_checkpoints:
        plot_training(args.training_checkpoints, args.output_dir / "training_returns.png", args.window)
    if args.evaluation:
        plot_evaluation(args.evaluation, args.output_dir / "evaluation_by_budget.png")
    if args.gru_final and args.mlp_final:
        plot_policies(args.gru_final, args.mlp_final, args.output_dir / "final_policy_maps.png")


if __name__ == "__main__":
    main()
