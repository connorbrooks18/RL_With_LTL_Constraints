"""Paired evaluation of q-visible GRU and current-observation MLP in Li's office."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from dqn import DQN, GRUDQN
from train_office_pomdp import START, build_environment


def _stats(samples):
    values = np.asarray(samples, dtype=float)
    mean = float(values.mean())
    se = float(values.std(ddof=1) / np.sqrt(values.size)) if values.size > 1 else 0.0
    return {"mean": mean, "ci95": [mean - 1.96 * se, mean + 1.96 * se], "n": int(values.size)}


def _load(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    metadata = checkpoint["metadata"]
    if metadata.get("architecture") == "gru":
        model = GRUDQN(
            int(metadata["n_observations"]), int(metadata["n_actions"]),
            hidden_size=int(metadata["gru_hidden_size"]),
            num_layers=int(metadata["gru_num_layers"]),
        )
    elif metadata.get("architecture") == "mlp":
        model = DQN(
            int(metadata["n_observations"]), int(metadata["n_actions"]),
            hidden_size=int(metadata["hidden_size"]),
        )
    else:
        raise ValueError(f"Unsupported architecture in {path}: {metadata.get('architecture')}")
    model.load_state_dict(checkpoint["policy_state_dict"])
    model.eval()
    return model, metadata


def _rollout(model, metadata, csrl, start, seed, horizon):
    np.random.seed(int(seed))
    transition_rng = np.random.default_rng(int(seed) + 1)
    state = (0, csrl.oa.q0, int(start[0]), int(start[1]))
    observations = [csrl.observe_state(state)]  # true q, analytic room readout
    previous_actions = [None]
    total_return = 0.0
    discount = 1.0
    accepting_cycles = 0
    storage_violation = "s" in tuple(csrl.mdp.label[state[2], state[3]])
    grid_action_count = len(csrl.mdp.A)
    history_length = int(metadata.get("history_length") or 4)

    for _ in range(0 if storage_violation else horizon):
        if metadata["architecture"] == "gru":
            encoded = csrl.encode_observation_history(
                observations,
                history_length=history_length,
                previous_actions=previous_actions,
                partial_observation=True,
                automaton_state_observed=True,
                epsilon_commitment_mode="none",
            ).unsqueeze(0)
        else:
            encoded = csrl.encode_observation(
                observations[-1],
                partial_observation=True,
                automaton_state_observed=True,
            ).unsqueeze(0)

        allowed = csrl.actions_for_observed_q(state[1])
        with torch.no_grad():
            q_values = model(encoded).squeeze(0).cpu().numpy()
        q_values[[a for a in range(len(q_values)) if a not in allowed]] = -np.inf
        action = int(np.argmax(q_values))

        reward = float(csrl.reward[state])
        if reward > 0:
            accepting_cycles += 1
        total_return += discount * reward
        discount *= csrl.discount_accepting if reward > 0 else csrl.discount_nonaccepting

        if action in csrl.A[state]:
            successors, probabilities = csrl.transition_probs[state][action]
            state = successors[transition_rng.choice(len(successors), p=probabilities)]
        else:
            raise RuntimeError(f"Invalid q-visible action {action} at {state}")

        if "s" in tuple(csrl.mdp.label[state[2], state[3]]):
            storage_violation = True
            break
        observations.append(csrl.observe_state(state))
        previous_actions.append(action)
        observations = observations[-history_length:]
        previous_actions = previous_actions[-history_length:]

    return {
        "discounted_return": total_return,
        "accepting_cycles": accepting_cycles,
        "safe_and_accepting": bool(accepting_cycles >= 1 and not storage_violation),
        "storage_violation": storage_violation,
    }


def _paired_rollout_summary(gru, mlp):
    result = {}
    fields = (
        ("discounted_return", "mean_discounted_return"),
        ("accepting_cycles", "mean_accepting_cycles"),
        ("safe_and_accepting", "safe_acceptance_rate"),
        ("storage_violation", "storage_violation_rate"),
    )
    for sample_key, output_key in fields:
        gru_values = [item[sample_key] for item in gru]
        mlp_values = [item[sample_key] for item in mlp]
        result[output_key] = {
            "gru": _stats(gru_values),
            "mlp": _stats(mlp_values),
            "paired_difference_gru_minus_mlp": _stats(
                np.asarray(gru_values, dtype=float) - np.asarray(mlp_values, dtype=float)
            ),
        }
    return result


def _evaluate_pair(gru_path, mlp_path, runs, horizon, seed):
    evaluation_started = datetime.now(timezone.utc)
    evaluation_started_perf = time.perf_counter()
    torch.set_num_threads(1)
    gru, gm = _load(gru_path)
    mlp, mm = _load(mlp_path)
    if (gm["seed"], gm["episodes"]) != (mm["seed"], mm["episodes"]):
        raise ValueError("GRU and MLP must be matched on training seed and episode count")
    for key in (
        "steps_per_episode", "learning_rate", "batch_size", "replay_capacity",
        "update_every", "target_update_every", "epsilon_floor",
        "epsilon_decay_fraction", "automaton_action_probability",
        "discount_nonaccepting", "discount_accepting", "start_mdp_square",
    ):
        if gm.get(key) != mm.get(key):
            raise ValueError(f"GRU and MLP checkpoints differ in {key}")
    if not gm.get("partial_observation") or not mm.get("partial_observation"):
        raise ValueError("Both policies must use partial grid observations")
    if not gm.get("automaton_state_observed") or not mm.get("automaton_state_observed"):
        raise ValueError("Both policies must receive the true automaton state q")

    _mdp, _oa, csrl = build_environment(
        float(gm["discount_nonaccepting"]), float(gm["discount_accepting"])
    )
    starts = [(row, col) for row in range(4) for col in range(4)]
    start_reports = []
    pooled_gru, pooled_mlp = [], []
    thesis_start_gru, thesis_start_mlp = [], []
    for start_index, start in enumerate(starts):
        gru_samples, mlp_samples = [], []
        for run in range(runs):
            evaluation_seed = int(seed) + start_index * 1_000_003 + run
            gru_samples.append(_rollout(gru, gm, csrl, start, evaluation_seed, horizon))
            mlp_samples.append(_rollout(mlp, mm, csrl, start, evaluation_seed, horizon))
        pooled_gru.extend(gru_samples)
        pooled_mlp.extend(mlp_samples)
        if start == START:
            thesis_start_gru = gru_samples
            thesis_start_mlp = mlp_samples
        start_reports.append({
            "square": list(start),
            "label": list(csrl.mdp.label[start]),
            "metrics": _paired_rollout_summary(gru_samples, mlp_samples),
        })

    return {
        "training_seed": int(gm["seed"]),
        "training_episodes": int(gm["episodes"]),
        "gru_training_wall_seconds": float(gm.get("elapsed_seconds", 0.0)),
        "mlp_training_wall_seconds": float(mm.get("elapsed_seconds", 0.0)),
        "gru_checkpoint": str(Path(gru_path).resolve()),
        "mlp_checkpoint": str(Path(mlp_path).resolve()),
        "history_length": int(gm["history_length"]),
        "q_state_observed": True,
        "grid_position_partially_observed": True,
        "observation_model": "full N,S,E,W room signature; uniform over cells sharing that signature",
        "start_position_used_for_training": list(START),
        "evaluation_rollouts_per_start": int(runs),
        "evaluation_horizon": int(horizon),
        "evaluation_seed": int(seed),
        "evaluation_started_at_utc": evaluation_started.isoformat(),
        "evaluation_finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_elapsed_seconds": time.perf_counter() - evaluation_started_perf,
        "metrics": _paired_rollout_summary(pooled_gru, pooled_mlp),
        "thesis_start_metrics": _paired_rollout_summary(thesis_start_gru, thesis_start_mlp),
        "starts": start_reports,
    }


def _t_critical(n):
    return {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571}.get(n, 1.96)


def _across_seeds(reports, metrics_key="metrics"):
    metrics = defaultdict(dict)
    fields = ("mean_discounted_return", "mean_accepting_cycles",
              "safe_acceptance_rate", "storage_violation_rate")
    for field in fields:
        by_seed = [report[metrics_key][field]["paired_difference_gru_minus_mlp"]["mean"]
                   for report in reports]
        values = np.asarray(by_seed, dtype=float)
        mean = float(values.mean())
        n = len(values)
        half = _t_critical(n) * float(values.std(ddof=1)) / np.sqrt(n) if n > 1 else 0.0
        metrics[field] = {
            "gru_minus_mlp_mean": mean,
            "ci95_across_training_seeds": [mean - half, mean + half],
            "n_training_seeds": n,
        }
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gru", type=Path, nargs="+", required=True)
    parser.add_argument("--mlp", type=Path, nargs="+", required=True)
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument("--horizon", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    def index(paths):
        result = {}
        for path in paths:
            metadata = torch.load(path, map_location="cpu", weights_only=True)["metadata"]
            key = (int(metadata["seed"]), int(metadata["episodes"]))
            if key in result:
                parser.error(f"duplicate checkpoint for seed/episodes {key}: {path}")
            result[key] = path
        return result

    gru_by_key, mlp_by_key = index(args.gru), index(args.mlp)
    if set(gru_by_key) != set(mlp_by_key):
        parser.error(f"GRU and MLP checkpoint cohorts differ: {set(gru_by_key)} vs {set(mlp_by_key)}")
    grouped = defaultdict(list)
    for seed, episodes in sorted(gru_by_key):
        print(f"Evaluating office pair seed={seed}, episodes={episodes} ...", flush=True)
        report = _evaluate_pair(
            gru_by_key[(seed, episodes)], mlp_by_key[(seed, episodes)],
            args.runs, args.horizon, args.seed,
        )
        grouped[episodes].append(report)
        metric = report["metrics"]["mean_discounted_return"]["paired_difference_gru_minus_mlp"]
        print(f"  paired return difference={metric['mean']:.5f} (rollout CI {metric['ci95']})", flush=True)

    by_episode = {
        str(episodes): {
            "training_seeds": [report["training_seed"] for report in reports],
            "across_training_seeds": _across_seeds(reports),
            "thesis_start_across_training_seeds": _across_seeds(
                reports, metrics_key="thesis_start_metrics"
            ),
            "runs": reports,
        }
        for episodes, reports in sorted(grouped.items())
    }
    output = {
        "experiment": "Junchao office: history-4 GRU-DQN versus current-observation MLP-DQN",
        "comparison_contract": (
            "Both receive true Rabin-pair and automaton q plus an analytically aliased room reading; "
            "only the GRU receives four observations and previous actions."
        ),
        "evaluation_design": "paired common-random-number rollouts from each of 16 rooms",
        "rollouts_per_start": int(args.runs),
        "horizon": int(args.horizon),
        "evaluation_seed": int(args.seed),
        "by_training_episodes": by_episode,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
