"""Compare POMDP GRU-DQN checkpoints with value-iteration policies.

The primary baseline indexes the full-state MDP value-iteration policy with
the noisy position observation and an automaton state inferred from the
controller's own epsilon action and perceived labels. An optional oracle
reports the MDP policy with access to the true product state for context.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np
import torch

from dqn import GRUDQN, RSSMDQN
from evaluate_pomdp import build_environment


def _rollout(
    csrl,
    start_square,
    rollout_seed,
    horizon,
    *,
    model=None,
    metadata=None,
    value_iteration_value=None,
    value_iteration_policy=None,
    oracle=False,
):
    np.random.seed(int(rollout_seed))
    transition_rng = np.random.default_rng(int(rollout_seed) + 1)
    state = (0, csrl.oa.q0, int(start_square[0]), int(start_square[1]))
    observation = csrl.observe_observation(state)
    observations = [observation]
    previous_actions = [None]
    inferred_q = csrl.oa.q0
    epsilon_target = None
    total_return = 0.0
    discount = 1.0
    grid_action_count = len(csrl.mdp.A)
    global_epsilon_actions = sorted({
        grid_action_count + target
        for outgoing in csrl.oa.eps
        for target in outgoing
    })
    global_actions = list(range(grid_action_count)) + global_epsilon_actions

    for _step in range(horizon):
        if model is not None:
            history = csrl.encode_observation_history(
                observations,
                history_length=int(metadata["history_length"]),
                previous_actions=previous_actions,
                partial_observation=True,
                epsilon_committed=epsilon_target is not None,
                epsilon_target=epsilon_target,
                epsilon_commitment_mode=metadata.get(
                    "recurrent_epsilon_commitment_mode", "target_one_hot"
                ),
            ).unsqueeze(0)
            with torch.no_grad():
                values = model(history).squeeze(0).cpu().numpy()
            allowed = (
                list(range(grid_action_count))
                if metadata.get("epsilon_commit_mask_from_history", False)
                and epsilon_target is not None
                else global_actions
            )
            values[[a for a in range(len(values)) if a not in allowed]] = -np.inf
            action = int(np.argmax(values))
        elif oracle:
            action = int(value_iteration_policy[state])
        else:
            observed_pair, observed_row, observed_col = observation
            estimated_state = (0, inferred_q, observed_row, observed_col)
            if epsilon_target is None:
                action = int(value_iteration_policy[estimated_state])
            else:
                # When committed, restrict the MDP greedy action to moves and
                # select by the same Bellman look-ahead that produced the policy.
                legal_grid = [
                    candidate for candidate in range(grid_action_count)
                    if candidate in csrl.A[estimated_state]
                ]
                action = max(
                    legal_grid,
                    key=lambda candidate: sum(
                        value_iteration_value[next_state] * probability
                        for next_state, probability in zip(
                            *csrl.transition_probs[estimated_state][candidate]
                        )
                    ),
                )

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

        if action >= grid_action_count and action in global_epsilon_actions:
            epsilon_target = action - grid_action_count
            inferred_q = epsilon_target
        elif not oracle:
            # Product transitions consume the label at the current cell before
            # moving. The observation-only baseline applies the same update to
            # its perceived current cell, which can be wrong under sensor noise.
            perceived_label = csrl.mdp.label[observation[1], observation[2]]
            inferred_q = csrl.oa.delta[inferred_q][perceived_label]

        observation = csrl.observe_observation(state)
        observations.append(observation)
        previous_actions.append(action)
        if model is not None:
            keep = int(metadata["history_length"])
            observations = observations[-keep:]
            previous_actions = previous_actions[-keep:]
    return total_return


def _summary(samples):
    samples = np.asarray(samples, dtype=float)
    mean = float(samples.mean())
    stderr = float(samples.std(ddof=1) / np.sqrt(samples.size)) if samples.size > 1 else 0.0
    return {
        "mean": mean,
        "ci95": [mean - 1.96 * stderr, mean + 1.96 * stderr],
        "n": int(samples.size),
    }


def _seed_aggregate(reports):
    """Aggregate only independent training seeds with matching checkpoints."""
    # 95% Student-t critical values for small seed counts (n=2..10).
    t_critical = {
        2: 12.706,
        3: 4.303,
        4: 3.182,
        5: 2.776,
        6: 2.571,
        7: 2.447,
        8: 2.365,
        9: 2.306,
        10: 2.262,
    }
    metric_names = (
        "mean_policy_dqn",
        "mean_mdp_value_iteration_on_noisy_observation",
        "paired_difference_policy_minus_vi",
        "mean_policy_dqn_non_c_starts",
        "mean_mdp_value_iteration_on_noisy_observation_non_c_starts",
        "paired_difference_policy_minus_vi_non_c_starts",
        "mean_mdp_value_iteration_full_state_oracle",
        "mean_mdp_value_iteration_full_state_oracle_non_c_starts",
    )
    groups = {}
    for report in reports:
        group_key = (
            report.get("architecture", "unknown"),
            int(report.get("training_episodes", -1)),
            report.get("training_phase", "unknown"),
        )
        groups.setdefault(group_key, []).append(report)
    aggregate = {}
    for (architecture, episodes, phase), group_reports in groups.items():
        group = {}
        for name in metric_names:
            entries = [report[name]["mean"] for report in group_reports
                       if report.get(name) is not None]
            if not entries:
                continue
            values = np.asarray(entries, dtype=float)
            mean = float(values.mean())
            if values.size == 1:
                half_width = 0.0
            else:
                critical = t_critical.get(int(values.size), 1.96)
                half_width = float(
                    critical * values.std(ddof=1) / np.sqrt(values.size)
                )
            group[name] = {
                "mean": mean,
                "ci95_across_training_seeds": [mean - half_width, mean + half_width],
                "n_training_seeds": int(values.size),
            }
        aggregate[f"{architecture}_K{episodes}_{phase}"] = group
    return aggregate


def evaluate_checkpoint(
    checkpoint_path,
    *,
    runs,
    horizon,
    seed,
    include_oracle,
    include_rollout_samples,
    vi_threshold,
):
    evaluation_started_at = datetime.now(timezone.utc)
    evaluation_started_perf = time.perf_counter()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    metadata = checkpoint["metadata"]
    policy_key = f"{metadata.get('architecture', 'gru')}_dqn"
    env = build_environment(float(metadata["observation_error"]))
    grid, oa, csrl, valid_starts, _eps, _actions, _sinks = env

    if metadata["architecture"] == "rssm":
        model = RSSMDQN(
            int(metadata["n_observations"]), int(metadata["n_actions"]),
            (csrl.shape[0], grid.shape[0], grid.shape[1]),
            hidden_size=int(metadata["rssm_deterministic_size"]),
            stochastic_variables=int(metadata["rssm_stochastic_variables"]),
            categorical_classes=int(metadata["rssm_categorical_classes"]),
        )
    else:
        model = GRUDQN(
            int(metadata["n_observations"]),
            int(metadata["n_actions"]),
            hidden_size=int(metadata["gru_hidden_size"]),
            num_layers=int(metadata["gru_num_layers"]),
        )
    model.load_state_dict(checkpoint["policy_state_dict"])
    model.eval()

    value = csrl.value_iteration(threshold=vi_threshold)
    vi_policy = csrl.greedy_policy(value)

    per_start = []
    all_dqn, all_vi, all_oracle = [], [], []
    non_c_dqn, non_c_vi, non_c_oracle = [], [], []
    for start_index, start_square in enumerate(valid_starts):
        dqn_returns, vi_returns, oracle_returns = [], [], []
        for rollout_index in range(runs):
            rollout_seed = seed + start_index * 1_000_003 + rollout_index
            dqn_return = _rollout(
                csrl, start_square, rollout_seed, horizon,
                model=model, metadata=metadata,
            )
            vi_return = _rollout(
                csrl, start_square, rollout_seed, horizon,
                value_iteration_value=value,
                value_iteration_policy=vi_policy,
            )
            dqn_returns.append(dqn_return)
            vi_returns.append(vi_return)
            if include_oracle:
                oracle_returns.append(_rollout(
                    csrl, start_square, rollout_seed, horizon,
                    value_iteration_policy=vi_policy, oracle=True,
                ))

        delta = np.asarray(dqn_returns) - np.asarray(vi_returns)
        item = {
            "square": [int(x) for x in start_square],
            "label": list(grid.label[start_square]),
            "policy_dqn": _summary(dqn_returns),
            "mdp_value_iteration_on_noisy_observation": _summary(vi_returns),
            "paired_difference_policy_minus_vi": _summary(delta),
        }
        item[policy_key] = item["policy_dqn"]
        item[f"paired_difference_{metadata.get('architecture', 'gru')}_minus_vi"] = item[
            "paired_difference_policy_minus_vi"
        ]
        if metadata.get("architecture", "gru") == "gru":
            item["paired_difference_gru_minus_vi"] = item[
                "paired_difference_policy_minus_vi"
            ]
        if include_rollout_samples:
            item["rollout_samples"] = {
                "policy_dqn": dqn_returns,
                policy_key: dqn_returns,
                "mdp_value_iteration_on_noisy_observation": vi_returns,
            }
        if include_oracle:
            item["mdp_value_iteration_full_state_oracle"] = _summary(oracle_returns)
            if include_rollout_samples:
                item["rollout_samples"][
                    "mdp_value_iteration_full_state_oracle"
                ] = oracle_returns
            all_oracle.extend(oracle_returns)
            if list(grid.label[start_square]) != ["c"]:
                non_c_oracle.extend(oracle_returns)
        per_start.append(item)
        all_dqn.extend(dqn_returns)
        all_vi.extend(vi_returns)
        if list(grid.label[start_square]) != ["c"]:
            non_c_dqn.extend(dqn_returns)
            non_c_vi.extend(vi_returns)

    paired = np.asarray(all_dqn) - np.asarray(all_vi)
    evaluation_finished_at = datetime.now(timezone.utc)
    return {
        "checkpoint": str(Path(checkpoint_path).resolve()),
        "run_id": metadata.get("run_id"),
        "history_length": int(metadata["history_length"]),
        "training_seed": metadata.get("seed"),
        "training_episodes": int(metadata.get("episodes", 0)),
        "training_phase": metadata.get("training_phase", "unknown"),
        "training_loop_seconds": float(metadata.get("training_loop_seconds", 0.0)),
        "gru_frozen_at_checkpoint": bool(
            metadata.get("gru_frozen_at_checkpoint", metadata.get("gru_frozen_at_end", False))
        ),
        "evaluation_started_at_utc": evaluation_started_at.isoformat(),
        "evaluation_finished_at_utc": evaluation_finished_at.isoformat(),
        "evaluation_elapsed_seconds": time.perf_counter() - evaluation_started_perf,
        "evaluation_seed": int(seed),
        "rollouts_per_start": int(runs),
        "rollout_samples_included": bool(include_rollout_samples),
        "horizon": int(horizon),
        "environment": {
            "shape": list(grid.shape),
            "transition_intended_probability": float(grid.p),
            "observation_error": float(grid.observation_error),
            "discount_nonaccepting": float(csrl.discount_nonaccepting),
            "discount_accepting": float(csrl.discount_accepting),
        },
        "value_iteration": {"stopping_threshold": vi_threshold},
        "mean_policy_dqn": _summary(all_dqn),
        "mean_mdp_value_iteration_on_noisy_observation": _summary(all_vi),
        "paired_difference_policy_minus_vi": _summary(paired),
        "mean_policy_dqn_non_c_starts": _summary(non_c_dqn),
        "mean_mdp_value_iteration_on_noisy_observation_non_c_starts": _summary(non_c_vi),
        "paired_difference_policy_minus_vi_non_c_starts": _summary(
            np.asarray(non_c_dqn) - np.asarray(non_c_vi)
        ),
        "mean_mdp_value_iteration_full_state_oracle": (
            _summary(all_oracle) if include_oracle else None
        ),
        "mean_mdp_value_iteration_full_state_oracle_non_c_starts": (
            _summary(non_c_oracle) if include_oracle else None
        ),
        "starts": per_start,
        "architecture": metadata.get("architecture", "gru"),
        "model_variant": (
            "RSSM-inspired fused recurrent categorical encoder; no stochastic feedback"
            if metadata.get("architecture") == "rssm" else "GRU-DQN"
        ),
        "baseline_definition": (
            "Exact product-MDP value iteration; at execution the policy is indexed "
            "by the noisy position reading and by an automaton state tracked from "
            "its own epsilon action and labels inferred from noisy positions."
        ),
        "oracle_definition": (
            "Same value-iteration policy with true product state at execution; "
            "reported only as a full-state reference."
            if include_oracle else None
        ),
    }
    report[f"mean_{policy_key}"] = report["mean_policy_dqn"]
    report[f"paired_difference_{metadata.get('architecture', 'gru')}_minus_vi"] = report[
        "paired_difference_policy_minus_vi"
    ]
    report[f"mean_{policy_key}_non_c_starts"] = report[
        "mean_policy_dqn_non_c_starts"
    ]
    if metadata.get("architecture", "gru") == "gru":
        report["mean_gru_dqn"] = report["mean_policy_dqn"]
        report["paired_difference_gru_minus_vi"] = report[
            "paired_difference_policy_minus_vi"
        ]
        report["mean_gru_dqn_non_c_starts"] = report[
            "mean_policy_dqn_non_c_starts"
        ]
        report["paired_difference_gru_minus_vi_non_c_starts"] = report[
            "paired_difference_policy_minus_vi_non_c_starts"
        ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoints", nargs="+", type=Path)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--horizon", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--vi-threshold", type=float, default=1e-8)
    parser.add_argument("--no-oracle", action="store_true")
    parser.add_argument(
        "--include-rollout-samples",
        action="store_true",
        help="include per-start return samples for paired policy comparisons",
    )
    parser.add_argument("--output", type=Path, default=Path(
        "artifacts/pomdp_history4_vs_mdp_value_iteration.json"
    ))
    args = parser.parse_args()

    batch_started_at = datetime.now(timezone.utc)
    batch_started_perf = time.perf_counter()
    reports = []
    for checkpoint in args.checkpoints:
        print(f"Evaluating {checkpoint} ...", flush=True)
        report = evaluate_checkpoint(
            checkpoint,
            runs=args.runs,
            horizon=args.horizon,
            seed=args.seed,
            include_oracle=not args.no_oracle,
            include_rollout_samples=args.include_rollout_samples,
            vi_threshold=args.vi_threshold,
        )
        reports.append(report)
        for name in (
            "mean_policy_dqn",
            "mean_mdp_value_iteration_on_noisy_observation",
            "paired_difference_policy_minus_vi",
            "mean_policy_dqn_non_c_starts",
            "mean_mdp_value_iteration_on_noisy_observation_non_c_starts",
            "paired_difference_policy_minus_vi_non_c_starts",
            "mean_mdp_value_iteration_full_state_oracle",
        ):
            result = report.get(name)
            if result is not None:
                lo, hi = result["ci95"]
                print(f"  {name}: {result['mean']:.6f} (95% CI {lo:.6f}, {hi:.6f})")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    across_seeds = _seed_aggregate(reports)
    batch_finished_at = datetime.now(timezone.utc)
    output = {
        "evaluation_batch": {
            "started_at_utc": batch_started_at.isoformat(),
            "finished_at_utc": batch_finished_at.isoformat(),
            "elapsed_seconds": time.perf_counter() - batch_started_perf,
            "runs": len(reports),
            "rollouts_per_start": args.runs,
            "horizon": args.horizon,
            "evaluation_seed": args.seed,
        },
        "runs": reports,
        "across_training_seeds": across_seeds,
    }
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    if len(reports) > 1:
        print("Across training seeds:")
        for group_name, metrics in across_seeds.items():
            print(f"  {group_name}:")
            for metric_name, result in metrics.items():
                lo, hi = result["ci95_across_training_seeds"]
                print(
                    f"    {metric_name}: {result['mean']:.6f} "
                    f"(95% CI {lo:.6f}, {hi:.6f}; "
                    f"n={result['n_training_seeds']})"
                )
    print(f"Saved comparison report: {args.output}", flush=True)


if __name__ == "__main__":
    main()
