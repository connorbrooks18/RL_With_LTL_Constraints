"""Paired greedy evaluation of GRU-DQN and feedforward DQN checkpoints."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path

import numpy as np
import torch

from dqn import DQN, GRUDQN
from evaluate_pomdp import build_environment


def _summary(values):
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    se = float(values.std(ddof=1) / np.sqrt(values.size)) if values.size > 1 else 0.0
    return {"mean": mean, "ci95": [mean - 1.96 * se, mean + 1.96 * se], "n": int(values.size)}


def _aggregate(runs):
    critical = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571,
                7: 2.447, 8: 2.365, 9: 2.306, 10: 2.262, 11: 2.228,
                12: 2.201, 13: 2.179, 14: 2.160, 15: 2.145, 16: 2.131,
                17: 2.120, 18: 2.110, 19: 2.101, 20: 2.093}
    result = {}
    for metric in ("mean_gru_dqn", "mean_no_gru_dqn", "paired_difference_gru_minus_no_gru",
                   "paired_difference_gru_minus_no_gru_non_c_starts"):
        values = np.asarray([run[metric]["mean"] for run in runs], dtype=float)
        mean = float(values.mean())
        half_width = (critical.get(values.size, 1.96) * values.std(ddof=1) / np.sqrt(values.size)
                      if values.size > 1 else 0.0)
        result[metric] = {
            "mean": mean,
            "ci95_across_training_seeds": [mean - float(half_width), mean + float(half_width)],
            "n_training_seeds": int(values.size),
        }
    return result


def _load(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    metadata = checkpoint["metadata"]
    architecture = metadata["architecture"]
    if architecture == "gru":
        model = GRUDQN(
            int(metadata["n_observations"]), int(metadata["n_actions"]),
            hidden_size=int(metadata["gru_hidden_size"]),
            num_layers=int(metadata["gru_num_layers"]),
        )
    elif architecture == "mlp":
        model = DQN(
            int(metadata["n_observations"]), int(metadata["n_actions"]),
            hidden_size=int(metadata["hidden_size"]),
        )
    else:
        raise ValueError(f"Expected a GRU or MLP checkpoint, got {architecture!r}: {path}")
    model.load_state_dict(checkpoint["policy_state_dict"])
    model.eval()
    return model, metadata


def _metadata(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    return checkpoint["metadata"]


def _rollout(csrl, start, rollout_seed, horizon, model, metadata):
    np.random.seed(int(rollout_seed))
    transition_rng = np.random.default_rng(int(rollout_seed) + 1)
    state = (0, csrl.oa.q0, int(start[0]), int(start[1]))
    observation = csrl.observe_observation(state)
    observations = [observation]
    previous_actions = [None]
    epsilon_target = None
    total_return = 0.0
    discount = 1.0
    grid_action_count = len(csrl.mdp.A)
    epsilon_actions = sorted({
        grid_action_count + target
        for outgoing in csrl.oa.eps
        for target in outgoing
    })
    global_actions = list(range(grid_action_count)) + epsilon_actions
    factors = (csrl.shape[0], csrl.mdp.shape[0], csrl.mdp.shape[1])

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
            allowed = (list(range(grid_action_count))
                       if metadata.get("epsilon_commit_mask_from_history", False)
                       and epsilon_target is not None else global_actions)
        else:
            encoded = torch.zeros(int(metadata["n_observations"]), dtype=torch.float32)
            offset = 0
            for coordinate, size in zip(observation, factors):
                encoded[offset + int(coordinate)] = 1.0
                offset += size
            encoded = encoded.unsqueeze(0)
            # Feedforward training uses the observation-independent global action mask.
            allowed = global_actions

        with torch.no_grad():
            q_values = model(encoded).squeeze(0).cpu().numpy()
        q_values[[action for action in range(len(q_values)) if action not in allowed]] = -np.inf
        action = int(np.argmax(q_values))

        reward = float(csrl.reward[state])
        total_return += discount * reward
        discount *= csrl.discount_accepting if reward > 0 else csrl.discount_nonaccepting

        if action in csrl.A[state]:
            successors, probabilities = csrl.transition_probs[state][action]
        elif action in epsilon_actions:
            successors, probabilities = [state], [1.0]
        else:
            raise RuntimeError(f"Invalid action {action} at {state}")
        state = successors[transition_rng.choice(len(successors), p=probabilities)]

        if metadata["architecture"] == "gru" and action in epsilon_actions and epsilon_target is None:
            epsilon_target = action - grid_action_count
        observation = csrl.observe_observation(state)
        if metadata["architecture"] == "gru":
            observations.append(observation)
            previous_actions.append(action)
            keep = int(metadata["history_length"])
            observations = observations[-keep:]
            previous_actions = previous_actions[-keep:]

    return total_return


def evaluate_pair(gru_path, no_gru_path, *, runs, horizon, seed):
    torch.set_num_threads(1)
    gru_model, gru_meta = _load(gru_path)
    no_gru_model, no_gru_meta = _load(no_gru_path)
    if gru_meta["seed"] != no_gru_meta["seed"]:
        raise ValueError("GRU and no-GRU checkpoints must use the same training seed")
    if gru_meta["environment_steps"] != no_gru_meta["environment_steps"]:
        raise ValueError("GRU and no-GRU checkpoints must have the same training step budget")
    if not gru_meta.get("partial_observation") or not no_gru_meta.get("partial_observation"):
        raise ValueError("Both checkpoints must hide the true product-state components")
    if gru_meta.get("observation_encoding") != no_gru_meta.get("observation_encoding"):
        raise ValueError("GRU and no-GRU checkpoints must use the same observation fields")
    if gru_meta.get("start_sampling") != no_gru_meta.get("start_sampling"):
        raise ValueError("GRU and no-GRU checkpoints must use the same episode start distribution")
    for key in ("episodes", "steps_per_episode", "learning_rate", "batch_size",
                "replay_capacity", "update_every", "target_update_every",
                "epsilon_floor", "epsilon_decay_fraction",
                "automaton_action_probability", "discount_nonaccepting",
                "discount_accepting"):
        if gru_meta.get(key) != no_gru_meta.get(key):
            raise ValueError(f"GRU and no-GRU checkpoints must match {key}")
    env = build_environment(float(gru_meta["observation_error"]))
    grid, _oa, csrl, starts, *_ = env
    if float(no_gru_meta["observation_error"]) != float(gru_meta["observation_error"]):
        raise ValueError("GRU and no-GRU checkpoints must use the same observation noise")

    per_start = []
    all_gru, all_no_gru, non_c_gru, non_c_no_gru = [], [], [], []
    for start_index, start in enumerate(starts):
        gru_returns, no_gru_returns = [], []
        for rollout_index in range(runs):
            rollout_seed = int(seed) + start_index * 1_000_003 + rollout_index
            gru_returns.append(_rollout(csrl, start, rollout_seed, horizon, gru_model, gru_meta))
            no_gru_returns.append(_rollout(csrl, start, rollout_seed, horizon, no_gru_model, no_gru_meta))
        difference = np.asarray(gru_returns) - np.asarray(no_gru_returns)
        per_start.append({
            "square": [int(x) for x in start],
            "label": list(grid.label[start]),
            "gru_dqn": _summary(gru_returns),
            "no_gru_dqn": _summary(no_gru_returns),
            "paired_difference_gru_minus_no_gru": _summary(difference),
            "rollout_samples": {"gru_dqn": gru_returns, "no_gru_dqn": no_gru_returns},
        })
        all_gru.extend(gru_returns)
        all_no_gru.extend(no_gru_returns)
        if list(grid.label[start]) != ["c"]:
            non_c_gru.extend(gru_returns)
            non_c_no_gru.extend(no_gru_returns)

    paired = np.asarray(all_gru) - np.asarray(all_no_gru)
    paired_non_c = np.asarray(non_c_gru) - np.asarray(non_c_no_gru)
    return {
        "training_seed": int(gru_meta["seed"]),
        "gru_checkpoint": str(Path(gru_path).resolve()),
        "no_gru_checkpoint": str(Path(no_gru_path).resolve()),
        "gru_episodes": int(gru_meta["episodes"]),
        "no_gru_episodes": int(no_gru_meta["episodes"]),
        "environment_steps": int(gru_meta["environment_steps"]),
        "history_length": int(gru_meta["history_length"]),
        "no_gru_architecture": no_gru_meta["architecture"],
        "evaluation_seed": int(seed),
        "rollouts_per_start": int(runs),
        "horizon": int(horizon),
        "mean_gru_dqn": _summary(all_gru),
        "mean_no_gru_dqn": _summary(all_no_gru),
        "paired_difference_gru_minus_no_gru": _summary(paired),
        "paired_difference_gru_minus_no_gru_non_c_starts": _summary(paired_non_c),
        "starts": per_start,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gru", type=Path, nargs="+", required=True)
    parser.add_argument("--no-gru", type=Path, nargs="+", required=True)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--horizon", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--workers", type=int, default=1,
                        help="Evaluate independent training-seed pairs concurrently.")
    parser.add_argument(
        "--training-seed", type=int,
        help="Evaluate only this matched training seed, for parallel per-seed runs.",
    )
    parser.add_argument("--output", type=Path, default=Path(
        "artifacts/pomdp_history4_vs_feedforward_dqn.json"
    ))
    args = parser.parse_args()
    gru_by_seed = {int(_metadata(path)["seed"]): path for path in args.gru}
    no_gru_by_seed = {int(_metadata(path)["seed"]): path for path in args.no_gru}
    if set(gru_by_seed) != set(no_gru_by_seed):
        parser.error(f"training seeds differ: GRU={sorted(gru_by_seed)}, no-GRU={sorted(no_gru_by_seed)}")

    if args.training_seed is not None:
        if args.training_seed not in gru_by_seed:
            parser.error(f"training seed {args.training_seed} is not present in both checkpoint sets")
        gru_by_seed = {args.training_seed: gru_by_seed[args.training_seed]}
        no_gru_by_seed = {args.training_seed: no_gru_by_seed[args.training_seed]}

    if args.workers < 1:
        parser.error("--workers must be at least 1")
    tasks = {
        training_seed: (gru_by_seed[training_seed], no_gru_by_seed[training_seed])
        for training_seed in sorted(gru_by_seed)
    }
    reports_by_seed = {}
    if args.workers == 1 or len(tasks) == 1:
        for training_seed, (gru_path, no_gru_path) in tasks.items():
            print(f"Evaluating matched training seed {training_seed} ...", flush=True)
            reports_by_seed[training_seed] = evaluate_pair(
                gru_path, no_gru_path, runs=args.runs,
                horizon=args.horizon, seed=args.seed,
            )
    else:
        with ProcessPoolExecutor(max_workers=min(args.workers, len(tasks))) as pool:
            futures = {
                pool.submit(
                    evaluate_pair, gru_path, no_gru_path,
                    runs=args.runs, horizon=args.horizon, seed=args.seed,
                ): training_seed
                for training_seed, (gru_path, no_gru_path) in tasks.items()
            }
            for future in as_completed(futures):
                training_seed = futures[future]
                reports_by_seed[training_seed] = future.result()
                result = reports_by_seed[training_seed]["paired_difference_gru_minus_no_gru"]
                lo, hi = result["ci95"]
                print(f"Seed {training_seed}: GRU − no-GRU {result['mean']:.6f} "
                      f"(95% rollout CI {lo:.6f}, {hi:.6f})", flush=True)
    reports = [reports_by_seed[seed] for seed in sorted(reports_by_seed)]

    output = {
        "comparison": "partial-observation GRU-DQN history-4 versus feedforward no-GRU DQN",
        "baseline_definition": "Feedforward MLP DQN receives only the current noisy (task-pair, row, column) observation. The task automaton state is hidden from both policies, and no observation history, previous action, or recurrent state is supplied to the MLP.",
        "runs": reports,
        "across_training_seeds": _aggregate(reports),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    aggregate = output["across_training_seeds"]["paired_difference_gru_minus_no_gru"]
    lo, hi = aggregate["ci95_across_training_seeds"]
    print(f"Across seeds: GRU minus no-GRU={aggregate['mean']:.6f} (95% CI {lo:.6f}, {hi:.6f})")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
