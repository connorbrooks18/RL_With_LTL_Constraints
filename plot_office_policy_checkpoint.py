"""Make notebook-style value/action and route maps for office checkpoints."""

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

from dqn import GRUDQN
from evaluate_pomdp import action_name
from office_pomdp import SIGNATURE_GROUP_BY_STATE, read_state_probabilities
from train_office_pomdp import START, build_environment


def load_policy(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    metadata = checkpoint["metadata"]
    model = GRUDQN(
        metadata["n_observations"], metadata["n_actions"],
        hidden_size=metadata["gru_hidden_size"],
        num_layers=metadata["gru_num_layers"],
    )
    model.load_state_dict(checkpoint["policy_state_dict"])
    model.eval()
    return model, metadata


def policy_map(model, metadata, mdp, oa, csrl, repeat_count):
    q_map = np.zeros(mdp.shape, dtype=float)
    greedy_action: dict[tuple[int, int], tuple[int, float]] = {}
    n_grid_actions = len(mdp.A)
    epsilon_actions = sorted({
        n_grid_actions + target for outgoing in oa.eps for target in outgoing
    })
    allowed = list(range(n_grid_actions)) + epsilon_actions
    for row in range(mdp.shape[0]):
        for col in range(mdp.shape[1]):
            state = (0, oa.q0, row, col)
            values: dict[int, float] = {}
            expected_max_q = 0.0
            for observed_position, probability in read_state_probabilities((row, col)).items():
                observation = (0, observed_position[0], observed_position[1])
                observations = [observation] * repeat_count
                previous_actions = [None] * repeat_count
                encoded = csrl.encode_observation_history(
                    observations,
                    history_length=int(metadata["history_length"]),
                    previous_actions=previous_actions,
                    partial_observation=True,
                    epsilon_commitment_mode=metadata.get(
                        "recurrent_epsilon_commitment_mode", "target_one_hot"
                    ),
                ).unsqueeze(0)
                with torch.no_grad():
                    q_values = model(encoded).squeeze(0).cpu().numpy()
                action = max(allowed, key=lambda candidate: q_values[candidate])
                values[action] = values.get(action, 0.0) + probability
                expected_max_q += probability * float(q_values[action])
            best_action = max(values, key=values.get)
            q_map[row, col] = expected_max_q
            greedy_action[(row, col)] = (best_action, values[best_action])
    return q_map, greedy_action


def draw_policy_map(ax, q_map, greedy_action, mdp, title, shared_min, shared_max):
    image = ax.imshow(q_map, cmap="viridis", interpolation="nearest",
                      vmin=shared_min, vmax=shared_max)
    ax.set_xticks(range(4))
    ax.set_yticks(range(4))
    ax.set_xticks(np.arange(-.5, 4, 1), minor=True)
    ax.set_yticks(np.arange(-.5, 4, 1), minor=True)
    ax.grid(which="minor", color="#eeeeee", linewidth=1.4)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.xaxis.tick_top()
    ax.set_title(title, fontsize=10)
    plt.colorbar(image, ax=ax, fraction=.046, pad=.04, label="Expected max Q")
    labels = {"a": "a", "c": "c", "p": "Print", "s": "S"}
    arrows = {"U": "↑", "D": "↓", "R": "→", "L": "←"}
    for row in range(4):
        for col in range(4):
            propositions = tuple(mdp.label[row, col])
            cell_label = labels.get(propositions[0], "") if propositions else ""
            if cell_label:
                ax.text(col, row - .28, cell_label, ha="center", va="center",
                        fontsize=7, fontweight="bold", color="#262626")
            action, probability = greedy_action[(row, col)]
            name = action_name(action, len(mdp.A))
            symbol = arrows.get(name, f"ε{action - len(mdp.A)}")
            if (row, col) == START:
                ax.scatter([col], [row], marker="o", s=100, facecolors="none",
                           edgecolors="#00a6a6", linewidths=2, zorder=4)
            ax.text(col, row + .10, symbol, ha="center", va="center",
                    fontsize=12, fontweight="bold",
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": .75, "pad": 1})
            alias_count = len(SIGNATURE_GROUP_BY_STATE[(row, col)])
            note = f"{probability:.1f}" if alias_count == 1 else f"×{alias_count} {probability:.1f}"
            ax.text(col + .31, row + .34, note, ha="center", va="center",
                    fontsize=6, color="#333333")
    return image


def plot_checkpoint(checkpoint_path, output_path):
    model, metadata = load_policy(checkpoint_path)
    mdp, oa, csrl = build_environment(
        discount_nonaccepting=float(metadata["discount_nonaccepting"]),
        discount_accepting=float(metadata["discount_accepting"]),
    )
    repeat_counts = [1, 4]
    figure, axes = plt.subplots(1, 2, figsize=(12, 5.7))
    panels = [policy_map(model, metadata, mdp, oa, csrl, count)
              for count in repeat_counts]
    shared_min = min(float(np.min(q_map)) for q_map, _actions in panels)
    shared_max = max(float(np.max(q_map)) for q_map, _actions in panels)
    if shared_max <= shared_min:
        shared_max = shared_min + 1e-6
    for ax, repeat_count, (q_map, actions) in zip(axes, repeat_counts, panels):
        panel = ("cold start (one observation)" if repeat_count == 1
                 else "four identical observations (history diagnostic)")
        draw_policy_map(ax, q_map, actions, mdp, panel, shared_min, shared_max)
    figure.suptitle(
        f"Office GRU-DQN · seed {metadata['seed']} · history {metadata['history_length']}\n"
        "Color = expected max Q under wall-signature aliases; arrow = modal greedy action; "
        "×n = alias-group size; ring = start",
        fontsize=11,
    )
    figure.tight_layout(rect=(0, 0, 1, .89))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(figure)
    return metadata


def plot_route(report_path, training_seed, output_path):
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    run = next(item for item in report["runs"] if item["training_seed"] == training_seed)
    route_data = run["sample_route_from_thesis_start"]
    route = route_data["positions"]
    phases = route_data["task_phase"]
    thesis_start = next(item for item in run["starts"]
                        if item["square"] == list(START))
    mdp, _oa, _csrl = build_environment()
    figure, ax = plt.subplots(figsize=(6.4, 6.1))
    ax.set_facecolor("#f3ead1")
    ax.set_xlim(-.5, 3.5)
    ax.set_ylim(3.5, -.5)
    ax.set_xticks(range(4))
    ax.set_yticks(range(4))
    ax.set_xticks(np.arange(-.5, 4, 1), minor=True)
    ax.set_yticks(np.arange(-.5, 4, 1), minor=True)
    ax.grid(which="minor", color="#333333", linewidth=1.4)
    ax.tick_params(which="minor", bottom=False, left=False)
    labels = {"a": "a", "c": "c", "p": "Print", "s": "S"}
    for row in range(4):
        for col in range(4):
            propositions = tuple(mdp.label[row, col])
            cell_label = labels.get(propositions[0], "") if propositions else ""
            if cell_label:
                ax.text(col, row - .20, cell_label, ha="center", va="center",
                        fontsize=9, fontweight="bold")
    coordinates = np.asarray(route)
    phase_colors = {0: "#137c8b", 1: "#d68a00"}
    for index in range(1, len(coordinates)):
        phase = int(phases[index])
        ax.plot(coordinates[index - 1:index + 1, 1],
                coordinates[index - 1:index + 1, 0],
                color=phase_colors[phase], linewidth=2,
                marker="o", markersize=2.4, alpha=.85)
    ax.scatter([START[1]], [START[0]], s=90, color="#00a6a6", edgecolor="white",
               linewidth=1.2, zorder=4, label="start")
    ax.set_title(
        f"Office rollout · training seed {training_seed}\n"
        f"{run['horizon']}-step horizon; "
        f"≥2 LTL cycles={thesis_start['at_least_two_ltl_cycles_without_storage_violation']['mean']:.2f}; "
        f"sample accepting cycles={route_data['completed_ltl_acceptance_cycles']}; "
        f"storage violation={route_data['storage_violation']}",
        fontsize=11,
    )
    handles = [
        Line2D([0], [0], color=phase_colors[0], linewidth=2,
               label="seeking printer"),
        Line2D([0], [0], color=phase_colors[1], linewidth=2,
               label="carrying to a/c"),
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=8)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evaluation-report", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    metadata = plot_checkpoint(args.checkpoint, args.output)
    print(f"Saved policy map {args.output} for seed {metadata['seed']}")
    if args.evaluation_report:
        route_output = args.output.with_name(args.output.stem + "_route.png")
        plot_route(args.evaluation_report, int(metadata["seed"]), route_output)
        print(f"Saved route map {route_output}")


if __name__ == "__main__":
    main()
