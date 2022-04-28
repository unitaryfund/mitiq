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

from re import I
import numpy as np
import cirq
from mitiq.ddd.insertion import (
    _get_circuit_mask,
    get_slack_matrix_from_circuit_mask,
    insert_ddd_sequences
)
import pytest
import qiskit
from mitiq.ddd.rules import xx, xyxy

circuit_cirq_one = cirq.Circuit(
    cirq.SWAP(q, q + 1) for q in cirq.LineQubit.range(7)
)
for i in (0,7):
    circuit_cirq_one.append(cirq.ops.I.on(i),
        cirq.ops.X.on(i),
        cirq.ops.Y.on(i),
        cirq.ops.X.on(i),
        cirq.ops.Y.on(i),
        cirq.ops.I.on(i),
    )
for i in (1,6):
    circuit_cirq_one.append(cirq.ops.I.on(i),
        cirq.ops.X.on(i),
        cirq.ops.Y.on(i),
        cirq.ops.X.on(i),
        cirq.ops.Y.on(i),
    )
for i in (2, 5):
    circuit_cirq_one.append(cirq.ops.X.on(i),
        cirq.ops.Y.on(i),
        cirq.ops.X.on(i),
        cirq.ops.Y.on(i),
    )

qreg_cirq = cirq.GridQubit.rect(10, 1)
circuit_cirq_two = cirq.Circuit(
    cirq.ops.H.on_each(*qreg_cirq), cirq.ops.H.on(qreg_cirq[1]),
)

circuit_cirq_three = cirq.Circuit(
    cirq.ops.H.on_each(*qreg_cirq), 5*[cirq.ops.H.on(qreg_cirq[1])],
)

circuit_cirq_three_validated = cirq.Circuit(
    cirq.ops.H.on_each(*qreg_cirq), 5*[cirq.ops.H.on(qreg_cirq[1])],
    cirq.ops.I.on_each(qreg_cirq[0], *qreg_cirq[2:]),
    cirq.ops.X.on_each(qreg_cirq[0], *qreg_cirq[2:]),
    cirq.ops.Y.on_each(qreg_cirq[0], *qreg_cirq[2:]),
    cirq.ops.X.on_each(qreg_cirq[0], *qreg_cirq[2:]),
    cirq.ops.Y.on_each(qreg_cirq[0], *qreg_cirq[2:])
)

qreg = qiskit.QuantumRegister(4)
circuit_qiskit_one = qiskit.QuantumCircuit(qreg)
for q in qreg:
    circuit_qiskit_one.x(q)
    circuit_qiskit_one.x(3)
circuit_qiskit_one.cx(0, 3)

qiskit.QuantumRegister(4)
circuit_qiskit_validate = qiskit.QuantumCircuit(qreg)
for i in range(4):
    circuit_qiskit_validate.x(i)
    circuit_qiskit_validate.x(3)
    if i != 3 and i != 0:
        circuit_qiskit_validate.i(i)
        circuit_qiskit_validate.x(i)
        circuit_qiskit_validate.i(i)
        circuit_qiskit_validate.x(i)
        circuit_qiskit_validate.i(i)
    elif i == 0:
        circuit_qiskit_validate.i(i)
        circuit_qiskit_validate.x(i)
        circuit_qiskit_validate.x(i)
        circuit_qiskit_validate.i(i)
circuit_qiskit_validate.cx(0, 3)

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


@pytest.mark.parametrize(
    ("circuit", "result", "rule"),
    [
        (circuit_cirq_one, test_mask_one, xyxy),
        (circuit_cirq_two, circuit_cirq_two, xx),
        (circuit_cirq_three, circuit_cirq_three_validated, xyxy),
        (circuit_qiskit_one, circuit_qiskit_validate, xx)
    ],

)
def test_insert_sequences(circuit, result, rule):
    circuit_with_sequences = insert_ddd_sequences(circuit, rule)
    assert circuit_with_sequences == result

