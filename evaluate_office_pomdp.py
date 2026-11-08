"""Evaluate trained history-4 GRU policies on the 4x4 office POMDP."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np
import torch

from dqn import GRUDQN
from train_office_pomdp import START, build_environment


def _mean_ci(values, *, binary=False):
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    if binary:
        n = int(values.size)
        z = 1.96
        denominator = 1.0 + z * z / n
        center = (mean + z * z / (2.0 * n)) / denominator
        half_width = z * np.sqrt(mean * (1.0 - mean) / n + z * z / (4.0 * n * n)) / denominator
        interval = [float(center - half_width), float(center + half_width)]
    else:
        se = float(values.std(ddof=1) / np.sqrt(values.size)) if values.size > 1 else 0.0
        interval = [mean - 1.96 * se, mean + 1.96 * se]
    return {"mean": mean, "ci95": interval, "n": int(values.size)}


def _across_training_seeds(reports):
    t_critical = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776,
                  6: 2.571, 7: 2.447, 8: 2.365, 9: 2.306, 10: 2.262,
                  11: 2.228, 12: 2.201, 13: 2.179, 14: 2.160,
                  15: 2.145, 16: 2.131, 17: 2.120, 18: 2.110,
                  19: 2.101, 20: 2.093}
    results = []
    for start_index in range(16):
        start_reports = [report["starts"][start_index] for report in reports]
        entry = {"square": start_reports[0]["square"],
                 "label": start_reports[0]["label"]}
        for metric in ("discounted_return", "completed_ltl_acceptance_cycles",
                       "at_least_one_ltl_cycle_without_storage_violation",
                       "at_least_two_ltl_cycles_without_storage_violation",
                       "completed_print_to_office_label_sequences",
                       "no_storage_violation"):
            values = np.asarray([item[metric]["mean"] for item in start_reports], dtype=float)
            mean = float(values.mean())
            n = int(values.size)
            half_width = (t_critical.get(n, 1.96) * values.std(ddof=1) / np.sqrt(n)
                          if n > 1 else 0.0)
            entry[metric] = {
                "mean": mean,
                "ci95_across_training_seeds": [mean - float(half_width), mean + float(half_width)],
                "n_training_seeds": n,
            }
        results.append(entry)
    return results


def load_policy(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    metadata = checkpoint["metadata"]
    if metadata["architecture"] != "gru":
        raise ValueError(f"Expected a GRU checkpoint, got {metadata['architecture']}")
    model = GRUDQN(
        metadata["n_observations"], metadata["n_actions"],
        hidden_size=metadata["gru_hidden_size"],
        num_layers=metadata["gru_num_layers"],
    )
    model.load_state_dict(checkpoint["policy_state_dict"])
    model.eval()
    return model, metadata


def rollout(model, metadata, mdp, oa, csrl, *, evaluation_seed, horizon,
            start_position=START, keep_trajectory=False):
    np.random.seed(int(evaluation_seed))
    transition_rng = np.random.default_rng(int(evaluation_seed) + 1)
    state = (0, oa.q0, int(start_position[0]), int(start_position[1]))
    observation = csrl.observe_observation(state)
    observations = [observation]
    previous_actions = [None]
    epsilon_target = None
    total_return = 0.0
    discount = 1.0
    mission_phase = 0
    deliveries = 0
    ltl_acceptance_cycles = 0
    storage_violation = False
    trajectory = [list(map(int, start_position))] if keep_trajectory else None
    trajectory_phases = [mission_phase] if keep_trajectory else None
    grid_action_count = len(mdp.A)
    epsilon_actions = sorted({
        grid_action_count + target
        for outgoing in oa.eps for target in outgoing
    })
    global_actions = list(range(grid_action_count)) + epsilon_actions
    history_length = int(metadata["history_length"])

    initial_labels = tuple(mdp.label[state[2], state[3]])
    if "p" in initial_labels:
        mission_phase = 1
    if "s" in initial_labels:
        storage_violation = True

    steps_taken = 0
    for step in range(horizon):
        if storage_violation:
            break
        steps_taken += 1
        encoded = csrl.encode_observation_history(
            observations,
            history_length=history_length,
            previous_actions=previous_actions,
            partial_observation=True,
            epsilon_committed=epsilon_target is not None,
            epsilon_target=epsilon_target,
            epsilon_commitment_mode=metadata.get(
                "recurrent_epsilon_commitment_mode", "target_one_hot"
            ),
        ).unsqueeze(0)
        allowed = (list(range(grid_action_count))
                   if metadata.get("epsilon_commit_mask_from_history", False)
                   and epsilon_target is not None else global_actions)
        with torch.no_grad():
            q_values = model(encoded).squeeze(0).cpu().numpy()
        q_values[[action for action in range(len(q_values)) if action not in allowed]] = -np.inf
        action = int(np.argmax(q_values))

        reward = float(csrl.reward[state])
        if reward > 0:
            # The LDBA acceptance reward is the authoritative task-cycle
            # signal. Position-only p→(a|c) counting is kept separately as a
            # diagnostic because epsilon commitment may select one branch.
            ltl_acceptance_cycles += 1
        total_return += discount * reward
        discount *= csrl.discount_accepting if reward > 0 else csrl.discount_nonaccepting
        if action in csrl.A[state]:
            successors, probabilities = csrl.transition_probs[state][action]
            state = successors[transition_rng.choice(len(successors), p=probabilities)]
        elif action in epsilon_actions:
            state = state
        else:
            raise RuntimeError(f"Invalid action {action} at product state {state}")

        if action in epsilon_actions and epsilon_target is None:
            epsilon_target = action - grid_action_count
        row, col = state[2], state[3]
        if trajectory is not None:
            trajectory.append([int(row), int(col)])
        labels = tuple(mdp.label[row, col])
        if "s" in labels:
            storage_violation = True
        if mission_phase == 0 and "p" in labels:
            mission_phase = 1
        elif mission_phase == 1 and ("a" in labels or "c" in labels):
            deliveries += 1
            mission_phase = 0
        if trajectory_phases is not None:
            trajectory_phases.append(mission_phase)
        if storage_violation:
            break

        observation = csrl.observe_observation(state)
        observations.append(observation)
        previous_actions.append(action)
        observations = observations[-history_length:]
        previous_actions = previous_actions[-history_length:]

    return {
        "discounted_return": float(total_return),
        "completed_ltl_acceptance_cycles": int(ltl_acceptance_cycles),
        "completed_print_to_office_label_sequences": int(deliveries),
        "storage_violation": bool(storage_violation),
        "steps": steps_taken,
        "trajectory": trajectory,
        "trajectory_task_phase": trajectory_phases,
    }


def evaluate_checkpoint(path, *, runs, horizon, evaluation_seed):
    evaluation_started_at = datetime.now(timezone.utc)
    evaluation_started_perf = time.perf_counter()
    torch.set_num_threads(1)
    model, metadata = load_policy(path)
    mdp, oa, csrl = build_environment(
        discount_nonaccepting=float(metadata["discount_nonaccepting"]),
        discount_accepting=float(metadata["discount_accepting"]),
    )
    starts = []
    representative_route = None
    for row in range(4):
        for col in range(4):
            start = (row, col)
            samples = [
                rollout(
                    model, metadata, mdp, oa, csrl,
                    evaluation_seed=evaluation_seed + start[0] * 100_003 + start[1] * 1_003 + run,
                    horizon=horizon,
                    start_position=start,
                    keep_trajectory=(start == START),
                )
                for run in range(runs)
            ]
            returns = [sample["discounted_return"] for sample in samples]
            cycle_counts = [sample["completed_ltl_acceptance_cycles"] for sample in samples]
            label_sequences = [sample["completed_print_to_office_label_sequences"]
                               for sample in samples]
            at_least_one = [count >= 1 and not sample["storage_violation"]
                            for count, sample in zip(cycle_counts, samples)]
            at_least_two = [count >= 2 and not sample["storage_violation"]
                            for count, sample in zip(cycle_counts, samples)]
            safe = [not sample["storage_violation"] for sample in samples]
            starts.append({
                "square": [row, col],
                "label": list(tuple(mdp.label[row, col])),
                "discounted_return": _mean_ci(returns),
                "completed_ltl_acceptance_cycles": _mean_ci(cycle_counts),
                "at_least_one_ltl_cycle_without_storage_violation": _mean_ci(at_least_one, binary=True),
                "at_least_two_ltl_cycles_without_storage_violation": _mean_ci(at_least_two, binary=True),
                "completed_print_to_office_label_sequences": _mean_ci(label_sequences),
                "no_storage_violation": _mean_ci(safe, binary=True),
            })
            if start == START:
                representative = next((sample for sample in samples
                                       if sample["completed_ltl_acceptance_cycles"] > 0),
                                      samples[0])
                representative_route = {
                    "positions": representative["trajectory"],
                    "task_phase": representative["trajectory_task_phase"],
                    "completed_ltl_acceptance_cycles": representative["completed_ltl_acceptance_cycles"],
                    "completed_print_to_office_label_sequences": representative["completed_print_to_office_label_sequences"],
                    "storage_violation": representative["storage_violation"],
                }
    report = {
        "training_seed": int(metadata["seed"]),
        "checkpoint": str(Path(path).resolve()),
        "episodes": int(metadata["episodes"]),
        "environment_steps": int(metadata["environment_steps"]),
        "history_length": int(metadata["history_length"]),
        "rollouts_per_start": int(runs),
        "horizon": int(horizon),
        "evaluation_seed": int(evaluation_seed),
        "thesis_start": list(START),
        "starts": starts,
        "sample_route_from_thesis_start": representative_route,
    }
    report["evaluation_started_at_utc"] = evaluation_started_at.isoformat()
    report["evaluation_finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    report["evaluation_elapsed_seconds"] = time.perf_counter() - evaluation_started_perf
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoints", type=Path, nargs="+", help="GRU checkpoint files")
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--horizon", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output", type=Path, default=Path("artifacts/office_pomdp_evaluation.json"))
    args = parser.parse_args()

    reports = []
    for path in args.checkpoints:
        print(f"Evaluating {path} ...", flush=True)
        report = evaluate_checkpoint(
            path, runs=args.runs, horizon=args.horizon, evaluation_seed=args.seed
        )
        reports.append(report)
        thesis_start = next(item for item in report["starts"]
                            if item["square"] == list(START))
        metric = thesis_start["at_least_two_ltl_cycles_without_storage_violation"]
        print(f"  thesis start: ≥2 LTL-accepting cycles {metric['mean']:.3f} "
              f"(95% rollout CI {metric['ci95'][0]:.3f}, {metric['ci95'][1]:.3f})", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "environment": "Li 4x4 single-agent office, full local wall signature with analytic alias probabilities",
        "task": "G F (p & F (a | c)) & G !s",
        "runs": reports,
        "across_training_seeds": _across_training_seeds(reports),
    }, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
