"""Compare history and automaton-state information on common POMDP rollouts."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np
import torch

from dqn import DQN, GRUDQN
from evaluate_pomdp import build_environment
from evaluate_pomdp_vs_value_iteration import _rollout as value_iteration_rollout


def summary(values: list[float]) -> dict:
    data = np.asarray(values, dtype=float)
    mean = float(data.mean())
    stderr = float(data.std(ddof=1) / np.sqrt(data.size)) if data.size > 1 else 0.0
    return {
        "mean": mean,
        "ci95_rollout_scenarios": [mean - 1.96 * stderr, mean + 1.96 * stderr],
        "n_rollout_scenarios": int(data.size),
    }


def paired_summary(values: list[float]) -> dict:
    result = summary(values)
    return {
        "mean_difference": result["mean"],
        "ci95_rollout_scenarios": result["ci95_rollout_scenarios"],
        "n_rollout_scenarios": result["n_rollout_scenarios"],
    }


def encode_one_hot(observation: tuple[int, ...], factors: tuple[int, ...]) -> torch.Tensor:
    encoded = torch.zeros(sum(factors), dtype=torch.float32)
    offset = 0
    for coordinate, size in zip(observation, factors):
        encoded[offset + int(coordinate)] = 1.0
        offset += size
    return encoded.unsqueeze(0)


def load_checkpoint(path: Path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    metadata = checkpoint["metadata"]
    if metadata["architecture"] == "gru":
        model = GRUDQN(
            int(metadata["n_observations"]), int(metadata["n_actions"]),
            hidden_size=int(metadata["gru_hidden_size"]),
            num_layers=int(metadata["gru_num_layers"]),
        )
    elif metadata["architecture"] == "mlp":
        model = DQN(
            int(metadata["n_observations"]), int(metadata["n_actions"]),
            hidden_size=int(metadata["hidden_size"]),
        )
    else:
        raise ValueError(f"Unsupported architecture {metadata['architecture']!r}")
    model.load_state_dict(checkpoint["policy_state_dict"])
    model.eval()
    return model, metadata


def learned_policy_rollout(csrl, grid, start, rollout_seed, horizon, model, metadata):
    np.random.seed(int(rollout_seed))
    transition_rng = np.random.default_rng(int(rollout_seed) + 1)
    state = (0, csrl.oa.q0, int(start[0]), int(start[1]))
    epsilon_target = None
    observations = [csrl.observe_observation(state)]
    previous_actions = [None]
    total_return = 0.0
    discount = 1.0
    grid_action_count = len(grid.A)
    epsilon_actions = sorted({
        grid_action_count + target
        for outgoing in csrl.oa.eps
        for target in outgoing
    })
    global_actions = list(range(grid_action_count)) + epsilon_actions
    partial = bool(metadata.get("partial_observation", True))
    automaton_state_observed = bool(metadata.get("automaton_state_observed", False))
    factors = (
        tuple(csrl.shape[:-1])
        if (not partial or automaton_state_observed)
        else (csrl.shape[0], grid.shape[0], grid.shape[1])
    )

    for _ in range(horizon):
        if metadata["architecture"] == "gru":
            encoded = csrl.encode_observation_history(
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
            allowed = (
                list(range(grid_action_count))
                if metadata.get("epsilon_commit_mask_from_history", False)
                and epsilon_target is not None else global_actions
            )
        else:
            observation = (
                csrl.observe_state(state)
                if (not partial or automaton_state_observed)
                else csrl.observe_observation(state)
            )
            encoded = encode_one_hot(observation, factors)
            allowed = global_actions if partial else list(csrl.A[state])

        with torch.no_grad():
            q_values = model(encoded).squeeze(0).cpu().numpy()
        q_values[[a for a in range(len(q_values)) if a not in allowed]] = -np.inf
        action = int(np.argmax(q_values))

        reward = float(csrl.reward[state])
        total_return += discount * reward
        discount *= csrl.discount_accepting if reward > 0 else csrl.discount_nonaccepting

        if action in csrl.A[state]:
            successors, probabilities = csrl.transition_probs[state][action]
        elif partial and action in epsilon_actions:
            # Under partial observation an unavailable epsilon attempt is a no-op.
            successors, probabilities = [state], [1.0]
        else:
            raise RuntimeError(f"Action {action} is invalid at product state {state}")
        state = successors[transition_rng.choice(len(successors), p=probabilities)]

        if metadata["architecture"] == "gru":
            if action in epsilon_actions and epsilon_target is None:
                epsilon_target = action - grid_action_count
            observations.append(csrl.observe_observation(state))
            previous_actions.append(action)
            keep = int(metadata["history_length"])
            observations = observations[-keep:]
            previous_actions = previous_actions[-keep:]

    return total_return


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gru", type=Path, required=True)
    parser.add_argument("--mlp-partial", type=Path, required=True)
    parser.add_argument("--mlp-product", type=Path, required=True,
                        help="MLP given true q, noisy position, and partial action interface")
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--horizon", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--vi-threshold", type=float, default=1e-8)
    parser.add_argument(
        "--output", type=Path,
        default=Path("artifacts/pomdp_automaton_state_ablation/evaluation.json"),
    )
    args = parser.parse_args()
    torch.set_num_threads(1)

    policies = {}
    for key, path in (
        ("history4_gru", args.gru),
        ("current_observation_mlp", args.mlp_partial),
        ("product_observation_mlp", args.mlp_product),
    ):
        model, metadata = load_checkpoint(path)
        policies[key] = {"model": model, "metadata": metadata, "path": str(path.resolve())}
    seeds = {int(value["metadata"]["seed"]) for value in policies.values()}
    if len(seeds) != 1:
        raise ValueError("All three learned policies must use the same training seed")
    if policies["current_observation_mlp"]["metadata"].get("partial_observation") is not True:
        raise ValueError("The current-observation MLP checkpoint must hide automaton state")
    product_metadata = policies["product_observation_mlp"]["metadata"]
    if (product_metadata.get("partial_observation") is not True
            or product_metadata.get("automaton_state_observed") is not True):
        raise ValueError("The product-observation MLP must see q and retain partial actions")

    observation_error = float(policies["history4_gru"]["metadata"]["observation_error"])
    grid, _oa, csrl, starts, _epsilon_actions, _global_actions, _sinks = (
        build_environment(observation_error)
    )
    vi_value = csrl.value_iteration(threshold=args.vi_threshold)
    vi_policy = csrl.greedy_policy(vi_value)
    all_returns = {
        key: [] for key in (
            *policies.keys(), "mdp_vi_noisy_input",
            "mdp_vi_q_observed_noisy_position", "mdp_vi_full_state_oracle",
        )
    }
    paired_values = {
        "product_mlp_minus_partial_mlp": [],
        "history4_gru_minus_product_mlp": [],
        "history4_gru_minus_noisy_input_vi": [],
        "product_mlp_minus_noisy_input_vi": [],
        "product_mlp_minus_q_observed_vi": [],
        "history4_gru_minus_q_observed_vi": [],
        "history4_gru_minus_full_state_vi": [],
        "product_mlp_minus_full_state_vi": [],
    }
    per_start = []
    started_at = datetime.now(timezone.utc)
    started_perf = time.perf_counter()

    for start_index, start in enumerate(starts):
        samples = {key: [] for key in all_returns}
        for rollout_index in range(args.runs):
            rollout_seed = args.seed + start_index * 1_000_003 + rollout_index
            for key, policy in policies.items():
                samples[key].append(learned_policy_rollout(
                    csrl, grid, start, rollout_seed, args.horizon,
                    policy["model"], policy["metadata"],
                ))
            samples["mdp_vi_noisy_input"].append(value_iteration_rollout(
                csrl, start, rollout_seed, args.horizon,
                value_iteration_value=vi_value,
                value_iteration_policy=vi_policy,
            ))
            samples["mdp_vi_q_observed_noisy_position"].append(value_iteration_rollout(
                csrl, start, rollout_seed, args.horizon,
                value_iteration_value=vi_value,
                value_iteration_policy=vi_policy,
                automaton_state_observed=True,
            ))
            samples["mdp_vi_full_state_oracle"].append(value_iteration_rollout(
                csrl, start, rollout_seed, args.horizon,
                value_iteration_policy=vi_policy,
                oracle=True,
            ))

        for key, values in samples.items():
            all_returns[key].extend(values)
        comparisons = {
            "product_mlp_minus_partial_mlp": np.asarray(samples["product_observation_mlp"])
            - np.asarray(samples["current_observation_mlp"]),
            "history4_gru_minus_product_mlp": np.asarray(samples["history4_gru"])
            - np.asarray(samples["product_observation_mlp"]),
            "history4_gru_minus_noisy_input_vi": np.asarray(samples["history4_gru"])
            - np.asarray(samples["mdp_vi_noisy_input"]),
            "product_mlp_minus_noisy_input_vi": np.asarray(samples["product_observation_mlp"])
            - np.asarray(samples["mdp_vi_noisy_input"]),
            "product_mlp_minus_q_observed_vi": np.asarray(samples["product_observation_mlp"])
            - np.asarray(samples["mdp_vi_q_observed_noisy_position"]),
            "history4_gru_minus_q_observed_vi": np.asarray(samples["history4_gru"])
            - np.asarray(samples["mdp_vi_q_observed_noisy_position"]),
            "history4_gru_minus_full_state_vi": np.asarray(samples["history4_gru"])
            - np.asarray(samples["mdp_vi_full_state_oracle"]),
            "product_mlp_minus_full_state_vi": np.asarray(samples["product_observation_mlp"])
            - np.asarray(samples["mdp_vi_full_state_oracle"]),
        }
        for key, values in comparisons.items():
            paired_values[key].extend(values.tolist())
        per_start.append({
            "square": [int(x) for x in start],
            "label": list(grid.label[start]),
            "policy_returns": {key: summary(values) for key, values in samples.items()},
            "paired_differences": {
                key: paired_summary(values.tolist())
                for key, values in comparisons.items()
            },
            "rollout_samples": samples,
        })
        print(f"Evaluated start {start_index + 1}/{len(starts)}: {tuple(start)}", flush=True)

    finished_at = datetime.now(timezone.utc)
    output = {
        "experiment": "automaton-state input ablation",
        "training_seed": next(iter(seeds)),
        "evaluation": {
            "started_at_utc": started_at.isoformat(),
            "finished_at_utc": finished_at.isoformat(),
            "elapsed_seconds": time.perf_counter() - started_perf,
            "evaluation_seed": args.seed,
            "rollouts_per_start": args.runs,
            "traversable_starts": len(starts),
            "horizon": args.horizon,
            "paired_scenarios_per_policy": len(starts) * args.runs,
            "position_observation_error": observation_error,
            "vi_threshold": args.vi_threshold,
        },
        "policies": {
            key: {
                "checkpoint": policy["path"],
                "architecture": policy["metadata"]["architecture"],
                "partial_observation": policy["metadata"].get("partial_observation"),
                "observation_encoding": policy["metadata"].get("observation_encoding"),
                "parameters": sum(p.numel() for p in policy["model"].parameters()),
                "mean_return": summary(all_returns[key]),
            }
            for key, policy in policies.items()
        },
        "value_iteration": {
            "mdp_vi_noisy_input": {
                "mean_return": summary(all_returns["mdp_vi_noisy_input"]),
                "description": "Exact product-MDP policy indexed by noisy position and its inferred automaton state.",
            },
            "mdp_vi_q_observed_noisy_position": {
                "mean_return": summary(all_returns["mdp_vi_q_observed_noisy_position"]),
                "description": "Exact product-MDP policy indexed by true automaton state and noisy position.",
            },
            "mdp_vi_full_state_oracle": {
                "mean_return": summary(all_returns["mdp_vi_full_state_oracle"]),
                "description": "Exact product-MDP policy given the true product state and true position.",
            },
        },
        "paired_differences": {
            key: paired_summary(values) for key, values in paired_values.items()
        },
        "starts": per_start,
        "interpretation_note": (
            "One matched training seed. Confidence intervals describe paired "
            "rollout-scenario variation only, not training-seed uncertainty. "
            "The product-observation MLP sees true automaton state, receives "
            "noisy physical position readings, and uses the same global action "
            "interface as the hidden-state policies."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {args.output}")
    for key, value in output["policies"].items():
        lo, hi = value["mean_return"]["ci95_rollout_scenarios"]
        print(f"{key}: {value['mean_return']['mean']:.6f} [{lo:.6f}, {hi:.6f}]")
    for key, value in output["value_iteration"].items():
        result = value["mean_return"]
        lo, hi = result["ci95_rollout_scenarios"]
        print(f"{key}: {result['mean']:.6f} [{lo:.6f}, {hi:.6f}]")
    for key, value in output["paired_differences"].items():
        lo, hi = value["ci95_rollout_scenarios"]
        print(f"{key}: {value['mean_difference']:.6f} [{lo:.6f}, {hi:.6f}]")


if __name__ == "__main__":
    main()
