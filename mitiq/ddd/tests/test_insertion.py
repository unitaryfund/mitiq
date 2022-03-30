# Copyright (C) 2022 Unitary Fund
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Unit tests for DDD slack windows and DDD insertion tools."""

import numpy as np
import cirq
from mitiq.ddd.insertion import (
    _get_circuit_mask,
    get_slack_matrix_from_circuit_mask,
)
import pytest

circuit_cirq_one = cirq.Circuit(
    cirq.SWAP(q, q + 1) for q in cirq.LineQubit.range(7)
)

qreg_cirq = cirq.GridQubit.rect(10, 1)
circuit_cirq_two = cirq.Circuit(
    cirq.ops.H.on_each(*qreg_cirq), cirq.ops.H.on(qreg_cirq[1])
)

# Define test mask matrices
test_mask_one = np.array(
    [
        [1, 0, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0, 0, 0],
        [0, 0, 1, 1, 0, 0, 0],
        [0, 0, 0, 1, 1, 0, 0],
        [0, 0, 0, 0, 1, 1, 0],
        [0, 0, 0, 0, 0, 1, 1],
        [0, 0, 0, 0, 0, 0, 1],
    ]
)

test_mask_two = np.array(
    [
        [1, 0],
        [1, 1],
        [1, 0],
        [1, 0],
        [1, 0],
        [1, 0],
        [1, 0],
        [1, 0],
        [1, 0],
        [1, 0],
    ]
)

one_mask = np.array(
    [
        [0, 1, 1, 1, 1],
        [1, 0, 1, 1, 1],
        [1, 1, 0, 1, 1],
        [1, 1, 1, 0, 1],
        [1, 1, 1, 1, 0],
        [0, 1, 0, 1, 1],
        [1, 0, 1, 0, 1],
        [0, 1, 0, 1, 0],
    ]
)
two_mask = np.array(
    [
        [0, 0, 1, 1, 1],
        [1, 0, 0, 1, 1],
        [1, 1, 0, 0, 1],
        [1, 1, 1, 0, 0],
        [0, 0, 1, 0, 0],
    ]
)
mixed_mask = np.array(
    [
        [1, 0, 1, 1, 1],
        [1, 1, 0, 0, 1],
        [0, 0, 0, 1, 1],
        [1, 1, 0, 0, 0],
        [1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1],
    ]
)
# Define test slack matrices
one_slack = np.array(
    [
        [1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1],
        [1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0],
        [1, 0, 1, 0, 1],
    ]
)
two_slack = np.array(
    [
        [2, 0, 0, 0, 0],
        [0, 2, 0, 0, 0],
        [0, 0, 2, 0, 0],
        [0, 0, 0, 2, 0],
        [2, 0, 0, 2, 0],
    ]
)
mixed_slack = np.array(
    [
        [0, 1, 0, 0, 0],
        [0, 0, 2, 0, 0],
        [3, 0, 0, 0, 0],
        [0, 0, 3, 0, 0],
        [0, 4, 0, 0, 0],
        [5, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ]
)
masks = [one_mask, two_mask, mixed_mask]
slack_matrices = [one_slack, two_slack, mixed_slack]


@pytest.mark.parametrize(
    ("circuit", "test_mask"),
    [(circuit_cirq_one, test_mask_one), (circuit_cirq_two, test_mask_two)],
)
def test_get_circuit_mask(circuit, test_mask):
    circuit_mask = _get_circuit_mask(circuit)
    assert np.allclose(circuit_mask, test_mask)


def test_get_slack_matrix_from_circuit_mask():
    for mask, expected in zip(masks, slack_matrices):
        slack_matrix = get_slack_matrix_from_circuit_mask(mask)
        assert np.allclose(slack_matrix, expected)


def test_get_slack_matrix_from_circuit_mask_extreme_cases():
    assert np.allclose(
        get_slack_matrix_from_circuit_mask(np.array([[0]])), np.array([[1]])
    )
    assert np.allclose(
        get_slack_matrix_from_circuit_mask(np.array([[1]])), np.array([[0]])
    )


def test_get_slack_matrix_from_circuit__bad_input_errors():
    with pytest.raises(TypeError, match="must be a numpy"):
        get_slack_matrix_from_circuit_mask([[1, 0], [0, 1]])
    with pytest.raises(ValueError, match="must be a 2-dimensional"):
        get_slack_matrix_from_circuit_mask(np.array([1, 2]))
    with pytest.raises(TypeError, match="must have integer elements"):
        get_slack_matrix_from_circuit_mask(np.array([[1, 0], [1, 1.7]]))
    with pytest.raises(ValueError, match="elements must be 0 or 1"):
        get_slack_matrix_from_circuit_mask(np.array([[2, 0], [0, 0]]))
