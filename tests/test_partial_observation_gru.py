import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

from controlsyn import ControlSynthesis
from dqn import GRUDQN
from mdp import GridMDP
from oa import OmegaAutomaton


def make_system(observation_error=0.0):
    structure = np.array([
        ["E", "E", "E", "E"],
        ["E", "E", "E", "T"],
        ["B", "E", "E", "E"],
        ["T", "E", "T", "E"],
        ["E", "E", "E", "E"],
    ])
    labels = np.array([
        [(), (), ("c",), ()],
        [(), (), ("a",), ("b",)],
        [(), (), ("c",), ()],
        [("b",), (), ("a",), ()],
        [(), ("c",), (), ("c",)],
    ], dtype=object)
    mdp = GridMDP(
        (5, 4), structure=structure, label=labels,
        observation_error=observation_error,
    )
    automaton = OmegaAutomaton("(F G a | F G b) & G !c", "ldba")
    return mdp, automaton, ControlSynthesis(mdp, automaton)


class ObservationTests(unittest.TestCase):
    def test_noisy_sensor_reports_neighbor_without_changing_environment(self):
        mdp = GridMDP((3, 3), observation_error=1.0)
        state = (1, 1)
        observation = mdp.observe_state(state)
        self.assertEqual(abs(observation[0] - state[0]) +
                         abs(observation[1] - state[1]), 1)
        self.assertEqual(state, (1, 1))
        with self.assertRaises(ValueError):
            mdp.observe_state(state, error_prob=1.1)

    def test_partial_observation_hides_automaton_state(self):
        _mdp, _oa, csrl = make_system()
        state_q0 = (0, 0, 1, 2)
        state_q2 = (0, 2, 1, 2)
        self.assertEqual(csrl.observe_observation(state_q0, error_prob=0),
                         (0, 1, 2))
        self.assertEqual(csrl.observe_observation(state_q0, error_prob=0),
                         csrl.observe_observation(state_q2, error_prob=0))

    def test_recurrent_history_is_left_padded_and_keeps_action_context(self):
        _mdp, _oa, csrl = make_system()
        history = csrl.encode_observation_history(
            [(0, 1, 2), (0, 1, 3)],
            history_length=3,
            previous_actions=[None, 5],
            epsilon_committed=True,
            epsilon_commitment_mode="target_one_hot",
        )
        self.assertEqual(tuple(history.shape), (3, 1 + 5 + 4 + 1 + 8 + 4))
        torch.testing.assert_close(history[0], torch.zeros_like(history[0]))
        self.assertEqual(float(history[1, 10]), 1.0)  # valid-token bit
        self.assertEqual(float(history[2, 10]), 1.0)
        self.assertEqual(float(history[2, 11 + 5]), 1.0)  # preceding action
        self.assertEqual(float(history[1, -3]), 1.0)  # remembered q1 target
        self.assertEqual(float(history[2, -3]), 1.0)


class GRUDQNTests(unittest.TestCase):
    def test_gru_maps_a_batch_of_histories_to_action_values(self):
        net = GRUDQN(n_observations=7, n_actions=6, hidden_size=8)
        inputs = torch.randn(4, 5, 7)
        outputs = net(inputs)
        self.assertEqual(tuple(outputs.shape), (4, 6))
        outputs.sum().backward()
        self.assertIsNotNone(net.gru.weight_ih_l0.grad)

    def test_training_uses_hidden_q_and_fixed_pomdp_action_set(self):
        torch.manual_seed(4)
        np.random.seed(4)
        mdp, oa, csrl = make_system(observation_error=0.25)
        with tempfile.TemporaryDirectory() as checkpoint_dir:
            Q, model = csrl.deep_q_learning(
                T=4,
                K=3,
                architecture="gru",
                history_length=3,
                gru_hidden_size=8,
                batch_size=2,
                replay_capacity=32,
                target_update_every=4,
                checkpoint_dir=checkpoint_dir,
                return_model=True,
            )
            metadata_path = next(Path(checkpoint_dir).glob("*.json"))
            metadata = json.loads(metadata_path.read_text())
            checkpoint_path = next(Path(checkpoint_dir).glob("*.pt"))
            checkpoint = torch.load(checkpoint_path, map_location="cpu",
                                    weights_only=True)
            self.assertEqual(metadata["architecture"], "gru")
            self.assertEqual(metadata["history_length"], 3)
            self.assertTrue(metadata["partial_observation"])
            self.assertTrue(metadata["epsilon_commit_mask_from_history"])
            self.assertTrue(metadata["recurrent_input_includes_epsilon_commitment"])
            self.assertEqual(metadata["training_return_summary"]["episodes"], 3)
            self.assertIn("zero bootstrap", metadata["rejecting_sink_target_semantics"])
            self.assertIn("policy_state_dict", checkpoint)
            self.assertEqual(tuple(checkpoint["episode_returns"].shape), (3,))

        self.assertEqual(csrl.last_episode_returns.shape, (3,))
        self.assertEqual(csrl.last_training_return_summary["episodes"], 3)
        self.assertEqual(Q.shape, csrl.shape)
        state_q0 = (0, oa.q0, 1, 2)
        state_q2 = (0, 2, 1, 2)
        np.testing.assert_allclose(Q[state_q0], Q[state_q2])
        self.assertTrue(np.isneginf(Q[state_q0][4]))
        self.assertTrue(np.isfinite(Q[state_q0][5]))
        self.assertTrue(np.isfinite(Q[state_q0][6]))

        observation = csrl.observe_observation(state_q0, error_prob=0.0)
        history = csrl.encode_observation_history(
            [observation],
            history_length=3,
            previous_actions=[None],
            epsilon_commitment_mode="target_one_hot",
        )
        self.assertEqual(tuple(model(history.unsqueeze(0)).shape),
                         (1, csrl.shape[-1]))

    def test_pomdp_rejecting_sink_stops_bootstrap_without_masking_actions(self):
        import dqn

        _mdp, _oa, csrl = make_system(observation_error=0.0)
        replay_instances = []
        base_replay_memory = dqn.ReplayMemory

        class CapturingReplayMemory(base_replay_memory):
            def __init__(self, capacity):
                super().__init__(capacity)
                replay_instances.append(self)

        # Starting on c and taking U sends q0 to the rejecting sink. Force U
        # so this checks the replay target deterministically.
        with mock.patch("dqn.ReplayMemory", CapturingReplayMemory), \
                mock.patch("random.random", side_effect=[0.5, 0.5]), \
                mock.patch("random.choice", return_value=0):
            csrl.deep_q_learning(
                start=(0, 2),
                T=1,
                K=1,
                architecture="gru",
                history_length=3,
                gru_hidden_size=8,
                partial_observation=True,
                automaton_action_probability=0.0,
                batch_size=2,
                replay_capacity=8,
                checkpoint_dir=None,
            )

        transition = replay_instances[0].memory[0]
        self.assertIsNone(transition.next_state)
        self.assertIsNone(transition.next_action_mask)

    def test_epsilon_commitment_masks_later_epsilon_actions_from_own_history(self):
        import dqn

        _mdp, _oa, csrl = make_system(observation_error=0.0)
        replay_instances = []
        base_replay_memory = dqn.ReplayMemory

        class CapturingReplayMemory(base_replay_memory):
            def __init__(self, capacity):
                super().__init__(capacity)
                replay_instances.append(self)

        # Commit q0 -> q1 on the first step, then take U. The action mask for
        # the post-commit history must contain only the four physical moves.
        with mock.patch("dqn.ReplayMemory", CapturingReplayMemory), \
                mock.patch("random.random", side_effect=[0.1, 0.5]), \
                mock.patch("random.choice", side_effect=[5, 0]):
            csrl.deep_q_learning(
                start=(0, 0),
                T=2,
                K=1,
                architecture="gru",
                history_length=3,
                gru_hidden_size=8,
                partial_observation=True,
                automaton_action_probability=1.0,
                batch_size=4,
                replay_capacity=8,
                checkpoint_dir=None,
            )

        committed_transition = replay_instances[0].memory[0]
        mask = committed_transition.next_action_mask
        self.assertTrue(bool(mask[:len(csrl.mdp.A)].all()))
        self.assertFalse(bool(mask[len(csrl.mdp.A):].any()))

    def test_noisy_feedforward_dqn_defaults_to_partial_observation(self):
        mdp, oa, csrl = make_system(observation_error=0.25)
        Q = csrl.deep_q_learning(
            T=4,
            K=2,
            architecture="mlp",
            batch_size=2,
            replay_capacity=32,
            target_update_every=4,
            checkpoint_dir=None,
        )
        np.testing.assert_allclose(Q[0, 0, 1, 2], Q[0, 2, 1, 2])
        self.assertTrue(np.isfinite(Q[0, oa.q0, 1, 2, 5]))
        self.assertTrue(np.isfinite(Q[0, 2, 1, 2, 5]))

    def test_exact_state_baseline_keeps_state_specific_action_mask(self):
        mdp, oa, csrl = make_system(observation_error=0.0)
        Q = csrl.deep_q_learning(
            T=4,
            K=2,
            architecture="linear",
            batch_size=2,
            replay_capacity=32,
            target_update_every=4,
            checkpoint_dir=None,
        )
        self.assertTrue(np.isfinite(Q[0, oa.q0, 1, 2, 5]))
        self.assertTrue(np.isneginf(Q[0, 2, 1, 2, 5]))


if __name__ == "__main__":
    unittest.main()
