"""Compare saved POMDP GRU-DQN checkpoints on matched greedy rollouts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from controlsyn import ControlSynthesis
from dqn import GRUDQN, RSSMDQN
from mdp import GridMDP
from oa import OmegaAutomaton


STRUCTURE = np.array([
    ["E", "E", "E", "E"],
    ["E", "E", "E", "T"],
    ["B", "E", "E", "E"],
    ["T", "E", "T", "E"],
    ["E", "E", "E", "E"],
])
LABEL = np.array([
    [(),       (),     ("c",), ()],
    [(),       (),     ("a",), ("b",)],
    [(),       (),     ("c",), ()],
    [("b",),  (),     ("a",), ()],
    [(),       ("c",), (),     ("c",)],
], dtype=object)


def action_name(action: int, grid_action_count: int) -> str:
    names = ("U", "D", "R", "L")
    if action < grid_action_count:
        return names[action]
    return f"eps->q{action - grid_action_count}"


def build_environment(observation_error: float):
    grid = GridMDP(
        shape=(5, 4),
        structure=STRUCTURE,
        label=LABEL,
        observation_error=observation_error,
    )
    oa = OmegaAutomaton("(F G a | F G b) & G !c", "ldba")
    csrl = ControlSynthesis(
        grid,
        oa,
        discount_nonaccepting=0.99999,
        discount_accepting=0.99,
    )
    valid_starts = [
        (int(r), int(c)) for r, c in grid.states()
        if grid.structure[r, c] != "B"
    ]
    epsilon_actions = sorted({
        len(grid.A) + target
        for outgoing in oa.eps
        for target in outgoing
    })
    global_actions = list(range(len(grid.A))) + epsilon_actions
    rejecting_sinks = {
        q for q in range(oa.shape[1])
        if not oa.eps[q]
        and oa.delta[q]
        and all(destination == q for destination in oa.delta[q].values())
        and not any(
            status is True
            for statuses in oa.acc[q].values()
            for status in statuses
        )
    }
    return grid, oa, csrl, valid_starts, epsilon_actions, global_actions, rejecting_sinks


def evaluate_checkpoint(path: Path, n_runs: int, horizon: int, seed: int):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    metadata = checkpoint["metadata"]
    grid, oa, csrl, valid_starts, epsilon_actions, global_actions, rejecting_sinks = (
        build_environment(float(metadata["observation_error"]))
    )
    if metadata["architecture"] == "rssm":
        model = RSSMDQN(
            metadata["n_observations"], metadata["n_actions"],
            (csrl.shape[0], grid.shape[0], grid.shape[1]),
            hidden_size=metadata["rssm_deterministic_size"],
            stochastic_variables=metadata["rssm_stochastic_variables"],
            categorical_classes=metadata["rssm_categorical_classes"],
        )
    else:
        model = GRUDQN(
            metadata["n_observations"],
            metadata["n_actions"],
            hidden_size=metadata["gru_hidden_size"],
            num_layers=metadata["gru_num_layers"],
        )
    model.load_state_dict(checkpoint["policy_state_dict"])
    model.eval()
    history_length = int(metadata["history_length"])
    commitment_mode = metadata.get("recurrent_epsilon_commitment_mode", "none")
    use_commit_mask = bool(metadata.get("epsilon_commit_mask_from_history", False))

    start_results = []
    for start_index, (start_row, start_col) in enumerate(valid_starts):
        np.random.seed(seed + start_index)
        transition_rng = np.random.default_rng(seed + start_index + 1)
        start_state = (0, oa.q0, start_row, start_col)
        returns = []
        first_actions = np.zeros(metadata["n_actions"], dtype=np.int64)
        marker_actions = {
            (int(row), int(col)): np.zeros(metadata["n_actions"], dtype=np.int64)
            for row, col in grid.states()
            if grid.label[row, col] in (("a",), ("b",))
        }

        for rollout in range(n_runs):
            state = start_state
            observations = [csrl.observe_observation(state)]
            previous_actions = [None]
            epsilon_target = None
            total_return = 0.0
            discount = 1.0

            for step in range(horizon):
                history = csrl.encode_observation_history(
                    observations,
                    history_length=history_length,
                    previous_actions=previous_actions,
                    partial_observation=True,
                    epsilon_committed=epsilon_target is not None,
                    epsilon_target=epsilon_target,
                    epsilon_commitment_mode=commitment_mode,
                ).unsqueeze(0)
                with torch.no_grad():
                    values = model(history).squeeze(0).cpu().numpy()
                allowed = (
                    list(range(len(grid.A)))
                    if use_commit_mask and epsilon_target is not None
                    else global_actions
                )
                values[[a for a in range(len(values)) if a not in allowed]] = -np.inf
                action = int(np.argmax(values))
                if step == 0:
                    first_actions[action] += 1
                if state[1] == oa.q0 and (state[2], state[3]) in marker_actions:
                    marker_actions[(state[2], state[3])][action] += 1

                reward = float(csrl.reward[state])
                total_return += discount * reward
                discount *= (
                    csrl.discount_accepting if reward > 0
                    else csrl.discount_nonaccepting
                )

                if action in csrl.A[state]:
                    successors, probabilities = csrl.transition_probs[state][action]
                else:
                    successors, probabilities = [state], [1.0]
                state = successors[
                    transition_rng.choice(len(successors), p=probabilities)
                ]
                if action in epsilon_actions and epsilon_target is None:
                    epsilon_target = action - len(grid.A)
                observations.append(csrl.observe_observation(state))
                previous_actions.append(action)
                observations = observations[-history_length:]
                previous_actions = previous_actions[-history_length:]
                if state[1] in rejecting_sinks:
                    break
            returns.append(total_return)

        samples = np.asarray(returns, dtype=float)
        mean = float(samples.mean())
        stderr = float(samples.std(ddof=1) / np.sqrt(n_runs)) if n_runs > 1 else 0.0
        start_results.append({
            "square": [start_row, start_col],
            "label": list(grid.label[start_row, start_col]),
            "mean_return": mean,
            "ci95": [mean - 1.96 * stderr, mean + 1.96 * stderr],
            "first_actions": first_actions.tolist(),
            "marker_actions": {
                f"{row},{col}": counts.tolist()
                for (row, col), counts in marker_actions.items()
            },
        })

    all_mean = float(np.mean([item["mean_return"] for item in start_results]))
    safe_starts = [item["mean_return"] for item in start_results if item["label"] != ["c"]]
    return {
        "checkpoint": str(path),
        "run_id": metadata.get("run_id"),
        "history_length": history_length,
        "seed": metadata.get("seed"),
        "evaluation_seed": seed,
        "rollouts_per_start": n_runs,
        "horizon": horizon,
        "mean_all_starts": all_mean,
        "mean_non_c_starts": float(np.mean(safe_starts)),
        "starts": start_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoints", nargs="+", type=Path)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--horizon", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    reports = []
    for checkpoint_path in args.checkpoints:
        print(f"Evaluating {checkpoint_path} ...", flush=True)
        report = evaluate_checkpoint(
            checkpoint_path, args.runs, args.horizon, args.seed
        )
        reports.append(report)
        print(
            f"history={report['history_length']} "
            f"mean(all)={report['mean_all_starts']:.5f} "
            f"mean(non-c)={report['mean_non_c_starts']:.5f}"
        )
        for item in report["starts"]:
            lo, hi = item["ci95"]
            print(
                f"  {tuple(item['square'])} label={item['label']} "
                f"mean={item['mean_return']:.5f} "
                f"CI95=({lo:.5f}, {hi:.5f})"
            )
            if item["label"] in (["a"], ["b"]):
                first = item["first_actions"]
                first_summary = ", ".join(
                    f"{action_name(i, 4)}={count}"
                    for i, count in enumerate(first) if count
                )
                row, col = item["square"]
                marker = item["marker_actions"][f"{row},{col}"]
                marker_summary = ", ".join(
                    f"{action_name(i, 4)}={count}"
                    for i, count in enumerate(marker) if count
                )
                print(f"    first actions: {first_summary}")
                print(f"    q0 actions on this marker: {marker_summary}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(reports, indent=2) + "\n")
        print("Saved matched rollout report:", args.output)


if __name__ == "__main__":
    main()
