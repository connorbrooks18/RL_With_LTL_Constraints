"""Plot a trained policy's noise-averaged cold-start action and Q maps."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

from dqn import DQN, GRUDQN, RSSMDQN
from evaluate_pomdp import action_name, build_environment


def load_policy(path: Path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    metadata = checkpoint["metadata"]
    if metadata["architecture"] == "gru":
        model = GRUDQN(
            metadata["n_observations"], metadata["n_actions"],
            hidden_size=metadata["gru_hidden_size"], num_layers=metadata["gru_num_layers"],
        )
    elif metadata["architecture"] == "rssm":
        grid, _oa, csrl, *_ = build_environment(float(metadata["observation_error"]))
        observation_factors = (
            tuple(csrl.shape[:-1])
            if metadata.get("automaton_state_observed", False)
            else (csrl.shape[0], grid.shape[0], grid.shape[1])
        )
        model = RSSMDQN(
            metadata["n_observations"], metadata["n_actions"],
            observation_factors,
            hidden_size=metadata["rssm_deterministic_size"],
            stochastic_variables=metadata["rssm_stochastic_variables"],
            categorical_classes=metadata["rssm_categorical_classes"],
        )
    elif metadata["architecture"] == "mlp":
        model = DQN(
            metadata["n_observations"], metadata["n_actions"],
            hidden_size=metadata["hidden_size"],
        )
    else:
        raise ValueError(f"Unsupported policy architecture: {metadata['architecture']}")
    model.load_state_dict(checkpoint["policy_state_dict"])
    model.eval()
    return model, metadata


def observation_distribution(mdp, position, error_probability):
    row, col = position
    neighbors = []
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        candidate = (row + dr, col + dc)
        if (0 <= candidate[0] < mdp.shape[0] and
                0 <= candidate[1] < mdp.shape[1] and
                mdp.structure[candidate] != "B"):
            neighbors.append(candidate)
    distribution = {tuple(position): 1.0 - error_probability}
    if neighbors:
        for neighbor in neighbors:
            distribution[neighbor] = distribution.get(neighbor, 0.0) + error_probability / len(neighbors)
    else:
        distribution[tuple(position)] = 1.0
    return distribution


def encoded_input(csrl, observation, model, metadata, repeat_count, q=None):
    if metadata["architecture"] in ("gru", "rssm"):
        if repeat_count == 1:
            observations = [observation]
            previous_actions = [None]
        else:
            observations = [observation] * repeat_count
            previous_actions = [None] * repeat_count
        return csrl.encode_observation_history(
            observations,
            history_length=int(metadata["history_length"]),
            previous_actions=previous_actions,
            partial_observation=True,
            automaton_state_observed=bool(
                metadata.get("automaton_state_observed", False)
            ),
            epsilon_commitment_mode=metadata.get("recurrent_epsilon_commitment_mode", "none"),
        ).unsqueeze(0)

    observes_q = (
        not metadata.get("partial_observation", True)
        or metadata.get("automaton_state_observed", False)
    )
    if observes_q:
        pair, _observed_q, row, col = observation
        factors = tuple(csrl.shape[:-1])
        if q is None:
            raise ValueError("A product-state observation requires an automaton state")
        coordinates = (pair, q, row, col)
    else:
        pair, row, col = observation
        factors = (csrl.shape[0], csrl.mdp.shape[0], csrl.mdp.shape[1])
        coordinates = (pair, row, col)
    vector = torch.zeros(int(metadata["n_observations"]), dtype=torch.float32)
    offset = 0
    for coordinate, size in zip(coordinates, factors):
        vector[offset + int(coordinate)] = 1.0
        offset += size
    return vector.unsqueeze(0)


def evaluate_panel(model, metadata, grid, oa, csrl, pair, repeat_count):
    q_map = np.full(grid.shape, np.nan, dtype=float)
    action_modes: dict[tuple[int, int], tuple[int, float]] = {}
    action_values: dict[tuple[int, int], dict[int, float]] = {}
    error_probability = float(metadata["observation_error"])
    grid_action_count = len(grid.A)
    epsilon_actions = sorted({
        grid_action_count + target
        for outgoing in oa.eps for target in outgoing
    })
    allowed_actions = list(range(grid_action_count)) + epsilon_actions
    observes_q = (
        not metadata.get("partial_observation", True)
        or metadata.get("automaton_state_observed", False)
    )
    for row, col in grid.states():
        if grid.structure[row, col] == "B":
            continue
        state = (pair, oa.q0, row, col)
        expected_q = 0.0
        action_probabilities: dict[int, float] = {}
        for observed_position, probability in observation_distribution(
                grid, (row, col), error_probability).items():
            observation = (
                (pair, oa.q0, observed_position[0], observed_position[1])
                if observes_q else
                (pair, observed_position[0], observed_position[1])
            )
            encoded = encoded_input(
                csrl, observation, model, metadata, repeat_count,
                q=oa.q0 if observes_q else None,
            )
            with torch.no_grad():
                q_values = model(encoded).squeeze(0).cpu().numpy()
            if not metadata.get("partial_observation", True):
                allowed = list(csrl.A[(pair, oa.q0, row, col)])
            elif observes_q:
                allowed = csrl.actions_for_observed_q(oa.q0)
            else:
                allowed = allowed_actions
            action = max(allowed, key=lambda candidate: q_values[candidate])
            expected_q += probability * float(q_values[action])
            action_probabilities[action] = action_probabilities.get(action, 0.0) + probability
        best_action = max(action_probabilities, key=action_probabilities.get)
        action_modes[(row, col)] = (best_action, action_probabilities[best_action])
        action_values[(row, col)] = action_probabilities
        q_map[row, col] = expected_q
    return q_map, action_modes, action_values


def draw_panel(ax, grid, q_map, action_modes, title, shared_limit=None):
    masked = np.ma.masked_invalid(q_map)
    cmap = grid.cmap.copy()
    cmap.set_bad("#777777")
    finite_values = q_map[np.isfinite(q_map)]
    limit = shared_limit
    if limit is None:
        limit = float(np.max(np.abs(finite_values))) if finite_values.size else 1.0
    limit = limit if limit > 0 else 1.0
    image = ax.imshow(masked, interpolation="nearest", cmap=cmap, vmin=-limit, vmax=limit)
    ax.set_xticks(np.arange(grid.shape[1]))
    ax.set_yticks(np.arange(grid.shape[0]))
    ax.set_xticks(np.arange(-.5, grid.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-.5, grid.shape[0], 1), minor=True)
    ax.grid(which="minor", color="lightgray", linestyle="-", linewidth=1, alpha=.7)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.xaxis.tick_top()
    ax.set_xlim(-.5, grid.shape[1] - .5)
    ax.set_ylim(grid.shape[0] - .5, -.5)
    plt.colorbar(image, ax=ax, fraction=.046, pad=.04, label="Expected max Q")
    ax.set_title(title, fontsize=10)
    arrows = {"U": "↑", "D": "↓", "R": "→", "L": "←"}
    for (row, col), (action, probability) in action_modes.items():
        labels = grid.label[row, col]
        cell_label = ",".join(labels) if labels else ("T" if grid.structure[row, col] == "T" else "")
        if cell_label:
            ax.text(col, row - .23, cell_label, ha="center", va="center",
                    fontsize=7, fontweight="bold", color="#252525")
        name = action_name(action, len(grid.A))
        symbol = arrows.get(name, rf"$\epsilon_{{{action - len(grid.A)}}}$")
        ax.text(
            col + 0.25, row + 0.27, symbol,
            ha="center", va="center", fontsize=11, fontweight="bold",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 1},
        )
        ax.text(col - 0.25, row + 0.29, f"{probability:.1f}",
                ha="center", va="center", fontsize=6, color="#222222",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.68, "pad": 0.5})
    for row, col in zip(*np.where(np.isnan(q_map))):
        ax.text(col, row, "B", ha="center", va="center", color="white",
                fontsize=10, fontweight="bold")


def plot_checkpoint(checkpoint_path: Path, output_path: Path, pair: int = 0):
    model, metadata = load_policy(checkpoint_path)
    grid, oa, csrl, *_ = build_environment(float(metadata["observation_error"]))
    repeats = [1, 4] if metadata["architecture"] in ("gru", "rssm") else [1]
    fig, axes = plt.subplots(1, len(repeats), figsize=(6.3 * len(repeats), 6.1), squeeze=False)
    panels = [evaluate_panel(model, metadata, grid, oa, csrl, pair, count)
              for count in repeats]
    finite_q = [q_map[np.isfinite(q_map)] for q_map, _actions, _probs in panels]
    shared_limit = max((float(np.max(np.abs(values))) for values in finite_q if values.size),
                       default=1.0)
    for ax, count, (q_map, action_modes, _probs) in zip(axes[0], repeats, panels):
        label = "cold start (one observation)" if count == 1 else "same observation repeated four times"
        if (metadata["architecture"] == "mlp"
                and metadata.get("automaton_state_observed", False)):
            label = "q₀ known; noisy physical position"
        draw_panel(ax, grid, q_map, action_modes,
                   f"{metadata['architecture'].upper()} policy: {label}", shared_limit)
    fig.suptitle(
        f"Seed {metadata['seed']} · history {metadata.get('history_length') or 0} · "
        f"noise {metadata['observation_error']:.2f}\n"
        "Color = expected max Q under sensor noise; arrow = most common greedy action; number = its probability",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pair", type=int, default=0)
    args = parser.parse_args()
    torch.set_num_threads(1)
    metadata = plot_checkpoint(args.checkpoint, args.output, args.pair)
    print(f"Saved {args.output} ({metadata['architecture']}, seed {metadata['seed']})")


if __name__ == "__main__":
    main()
