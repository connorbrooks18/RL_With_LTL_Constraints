"""Four-by-four office POMDP adapted from Li (2024), Figure 11."""

from __future__ import annotations

import numpy as np

from mdp import GridMDP


SHAPE = (4, 4)
# Direction order is north, south, east, west. Categories are the same as in
# Li's office model: hallway=0, wall=1, door=2, window=3.
WALL_SIGNATURES = np.array([
    [[1, 1, 2, 3], [1, 0, 2, 2], [1, 1, 2, 2], [1, 2, 1, 2]],
    [[1, 1, 2, 1], [0, 0, 2, 2], [1, 1, 1, 2], [2, 1, 1, 1]],
    [[1, 1, 2, 1], [0, 0, 1, 2], [1, 1, 2, 1], [1, 0, 1, 2]],
    [[1, 1, 2, 3], [0, 1, 0, 2], [1, 1, 0, 0], [0, 1, 1, 0]],
], dtype=np.int8)

LABEL = np.empty(SHAPE, dtype=object)
LABEL.fill(())
LABEL[0, 0] = ("a",)
LABEL[1, 2] = ("s",)
LABEL[2, 0] = ("c",)
LABEL[2, 2] = ("p",)

STRUCTURE = np.full(SHAPE, "E")
DIRECTIONS = ((-1, 0), (1, 0), (0, 1), (0, -1))
ACCESSIBLE_FEATURES = {0, 2}  # hallway or doorway


def signature_groups():
    groups: dict[tuple[int, ...], list[tuple[int, int]]] = {}
    for row in range(SHAPE[0]):
        for col in range(SHAPE[1]):
            signature = tuple(int(x) for x in WALL_SIGNATURES[row, col])
            groups.setdefault(signature, []).append((row, col))
    return groups


SIGNATURE_GROUPS = signature_groups()
SIGNATURE_GROUP_BY_STATE = {
    (row, col): tuple(cells)
    for cells in SIGNATURE_GROUPS.values()
    for row, col in cells
}


def read_state_probabilities(state):
    """Exact P(read_state=o | true state=s), derived from wall aliases.

    A four-direction local signature identifies a state exactly when that
    signature is unique. If several states share it, the readout returns one
    of those state IDs uniformly. No direction is sampled or hidden.
    """
    aliases = SIGNATURE_GROUP_BY_STATE[tuple(state)]
    probability = 1.0 / len(aliases)
    return {candidate: probability for candidate in aliases}


def read_state_matrix():
    """Return the 16x16 analytic state-readout matrix in row-major order."""
    matrix = np.zeros((SHAPE[0] * SHAPE[1], SHAPE[0] * SHAPE[1]), dtype=float)
    for row in range(SHAPE[0]):
        for col in range(SHAPE[1]):
            source = row * SHAPE[1] + col
            for (observed_row, observed_col), probability in read_state_probabilities((row, col)).items():
                observed = observed_row * SHAPE[1] + observed_col
                matrix[source, observed] = probability
    return matrix


class OfficeGridMDP(GridMDP):
    """Li office layout with signature-based position readout and motion."""

    def __init__(self, **kwargs):
        kwargs.setdefault("shape", SHAPE)
        kwargs.setdefault("structure", STRUCTURE.copy())
        kwargs.setdefault("label", LABEL.copy())
        kwargs.setdefault("p", 0.9)
        super().__init__(**kwargs)

    def read_state(self, state):
        """Return the true cell if architecturally unique, else an alias cell."""
        row, col = map(int, state)
        candidates = SIGNATURE_GROUP_BY_STATE[(row, col)]
        return candidates[int(np.random.randint(len(candidates)))]

    def observe_state(self, state, error_prob=None):
        # Keep GridMDP's observation interface while replacing its coordinate
        # noise with the analytically derived office signature readout.
        return self.read_state(state)

    def get_transition_prob(self, state, action_name):
        """Use the thesis's 0.9 intended-motion rule and office door layout."""
        row, col = map(int, state)
        open_directions = [
            direction for direction, category in enumerate(WALL_SIGNATURES[row, col])
            if int(category) in ACCESSIBLE_FEATURES
        ]
        if not open_directions:
            return [state], np.asarray([1.0])

        intended = self.A.index(action_name)
        if intended in open_directions:
            if len(open_directions) == 1:
                directions, probabilities = [intended], [1.0]
            else:
                directions = open_directions
                probabilities = [
                    self.p if direction == intended else
                    (1.0 - self.p) / (len(open_directions) - 1)
                    for direction in directions
                ]
            destinations = [
                (row + DIRECTIONS[direction][0], col + DIRECTIONS[direction][1])
                for direction in directions
            ]
            return destinations, np.asarray(probabilities, dtype=float)

        destinations = [tuple(state)] + [
            (row + DIRECTIONS[direction][0], col + DIRECTIONS[direction][1])
            for direction in open_directions
        ]
        probabilities = [self.p] + [
            (1.0 - self.p) / len(open_directions)
            for _ in open_directions
        ]
        return destinations, np.asarray(probabilities, dtype=float)
