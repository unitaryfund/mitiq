# Copyright (C) 2020 Unitary Fund
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

"""Tests for utility functions."""
from copy import deepcopy
import pytest

import numpy as np

import cirq
from cirq import (
    LineQubit,
    Circuit,
    X,
    Y,
    Z,
    H,
    CNOT,
    S,
    T,
    MeasurementGate,
    depolarize,
)

from mitiq.utils import (
    _are_close_dict,
    _equal,
    _simplify_gate_exponent,
    _simplify_circuit_exponents,
    _max_ent_state_circuit,
    _circuit_to_choi,
    _operation_to_choi,
)


@pytest.mark.parametrize("require_qubit_equality", [True, False])
def test_circuit_equality_identical_qubits(require_qubit_equality):
    qreg = cirq.NamedQubit.range(5, prefix="q_")
    circA = cirq.Circuit(cirq.ops.H.on_each(*qreg))
    circB = cirq.Circuit(cirq.ops.H.on_each(*qreg))
    assert circA is not circB
    assert _equal(circA, circB, require_qubit_equality=require_qubit_equality)


@pytest.mark.parametrize("require_qubit_equality", [True, False])
def test_circuit_equality_nonidentical_but_equal_qubits(
    require_qubit_equality,
):
    n = 5
    qregA = cirq.NamedQubit.range(n, prefix="q_")
    qregB = cirq.NamedQubit.range(n, prefix="q_")
    circA = cirq.Circuit(cirq.ops.H.on_each(*qregA))
    circB = cirq.Circuit(cirq.ops.H.on_each(*qregB))
    assert circA is not circB
    assert _equal(circA, circB, require_qubit_equality=require_qubit_equality)


def test_circuit_equality_linequbit_gridqubit_equal_indices():
    n = 10
    qregA = cirq.LineQubit.range(n)
    qregB = [cirq.GridQubit(x, 0) for x in range(n)]
    circA = cirq.Circuit(cirq.ops.H.on_each(*qregA))
    circB = cirq.Circuit(cirq.ops.H.on_each(*qregB))
    assert _equal(circA, circB, require_qubit_equality=False)
    assert not _equal(circA, circB, require_qubit_equality=True)


def test_circuit_equality_linequbit_gridqubit_unequal_indices():
    n = 10
    qregA = cirq.LineQubit.range(n)
    qregB = [cirq.GridQubit(x + 3, 0) for x in range(n)]
    circA = cirq.Circuit(cirq.ops.H.on_each(*qregA))
    circB = cirq.Circuit(cirq.ops.H.on_each(*qregB))
    assert _equal(circA, circB, require_qubit_equality=False)
    assert not _equal(circA, circB, require_qubit_equality=True)


def test_circuit_equality_linequbit_namedqubit_equal_indices():
    n = 8
    qregA = cirq.LineQubit.range(n)
    qregB = cirq.NamedQubit.range(n, prefix="q_")
    circA = cirq.Circuit(cirq.ops.H.on_each(*qregA))
    circB = cirq.Circuit(cirq.ops.H.on_each(*qregB))
    assert _equal(circA, circB, require_qubit_equality=False)
    assert not _equal(circA, circB, require_qubit_equality=True)


def test_circuit_equality_linequbit_namedqubit_unequal_indices():
    n = 11
    qregA = cirq.LineQubit.range(n)
    qregB = [cirq.NamedQubit(str(x + 10)) for x in range(n)]
    circA = cirq.Circuit(cirq.ops.H.on_each(*qregA))
    circB = cirq.Circuit(cirq.ops.H.on_each(*qregB))
    assert _equal(circA, circB, require_qubit_equality=False)
    assert not _equal(circA, circB, require_qubit_equality=True)


def test_circuit_equality_gridqubit_namedqubit_equal_indices():
    n = 8
    qregA = [cirq.GridQubit(0, x) for x in range(n)]
    qregB = cirq.NamedQubit.range(n, prefix="q_")
    circA = cirq.Circuit(cirq.ops.H.on_each(*qregA))
    circB = cirq.Circuit(cirq.ops.H.on_each(*qregB))
    assert _equal(circA, circB, require_qubit_equality=False)
    assert not _equal(circA, circB, require_qubit_equality=True)


def test_circuit_equality_gridqubit_namedqubit_unequal_indices():
    n = 5
    qregA = [cirq.GridQubit(x + 2, 0) for x in range(n)]
    qregB = [cirq.NamedQubit(str(x + 10)) for x in range(n)]
    circA = cirq.Circuit(cirq.ops.H.on_each(*qregA))
    circB = cirq.Circuit(cirq.ops.H.on_each(*qregB))
    assert _equal(circA, circB, require_qubit_equality=False)
    assert not _equal(circA, circB, require_qubit_equality=True)


def test_circuit_equality_unequal_measurement_keys_terminal_measurements():
    base_circuit = cirq.testing.random_circuit(
        qubits=5, n_moments=10, op_density=0.99, random_state=1
    )
    qreg = list(base_circuit.all_qubits())

    circ1 = deepcopy(base_circuit)
    circ1.append(cirq.measure(q, key="one") for q in qreg)

    circ2 = deepcopy(base_circuit)
    circ2.append(cirq.measure(q, key="two") for q in qreg)

    assert _equal(circ1, circ2, require_measurement_equality=False)
    assert not _equal(circ1, circ2, require_measurement_equality=True)


@pytest.mark.parametrize("require_measurement_equality", [True, False])
def test_circuit_equality_equal_measurement_keys_terminal_measurements(
    require_measurement_equality,
):
    base_circuit = cirq.testing.random_circuit(
        qubits=5, n_moments=10, op_density=0.99, random_state=1
    )
    qreg = list(base_circuit.all_qubits())

    circ1 = deepcopy(base_circuit)
    circ1.append(cirq.measure(q, key="z") for q in qreg)

    circ2 = deepcopy(base_circuit)
    circ2.append(cirq.measure(q, key="z") for q in qreg)

    assert _equal(
        circ1, circ2, require_measurement_equality=require_measurement_equality
    )


def test_circuit_equality_unequal_measurement_keys_nonterminal_measurements():
    base_circuit = cirq.testing.random_circuit(
        qubits=5, n_moments=10, op_density=0.99, random_state=1
    )
    end_circuit = cirq.testing.random_circuit(
        qubits=5, n_moments=5, op_density=0.99, random_state=2
    )
    qreg = list(base_circuit.all_qubits())

    circ1 = deepcopy(base_circuit)
    circ1.append(cirq.measure(q, key="one") for q in qreg)
    circ1 += end_circuit

    circ2 = deepcopy(base_circuit)
    circ2.append(cirq.measure(q, key="two") for q in qreg)
    circ2 += end_circuit

    assert _equal(circ1, circ2, require_measurement_equality=False)
    assert not _equal(circ1, circ2, require_measurement_equality=True)


@pytest.mark.parametrize("require_measurement_equality", [True, False])
def test_circuit_equality_equal_measurement_keys_nonterminal_measurements(
    require_measurement_equality,
):
    base_circuit = cirq.testing.random_circuit(
        qubits=5, n_moments=10, op_density=0.99, random_state=1
    )
    end_circuit = cirq.testing.random_circuit(
        qubits=5, n_moments=5, op_density=0.99, random_state=2
    )
    qreg = list(base_circuit.all_qubits())

    circ1 = deepcopy(base_circuit)
    circ1.append(cirq.measure(q, key="z") for q in qreg)
    circ1 += end_circuit

    circ2 = deepcopy(base_circuit)
    circ2.append(cirq.measure(q, key="z") for q in qreg)
    circ2 += end_circuit

    assert _equal(
        circ1, circ2, require_measurement_equality=require_measurement_equality
    )


@pytest.mark.parametrize("gate", [X ** 3, Y ** -3, Z ** -1, H ** -1])
def test_simplify_gate_exponent(gate):
    # Check exponent is simplified to 1
    assert _simplify_gate_exponent(gate).exponent == 1
    # Check simplified gate is equivalent to the input
    assert _simplify_gate_exponent(gate) == gate


@pytest.mark.parametrize("gate", [T ** -1, S ** -1, MeasurementGate(1)])
def test_simplify_gate_exponent_with_gates_that_cannot_be_simplified(gate):
    # Check the gate is not simplified (same representation)
    assert _simplify_gate_exponent(gate).__repr__() == gate.__repr__()


def test_simplify_circuit_exponents():
    qreg = LineQubit.range(2)
    circuit = Circuit([H.on(qreg[0]), CNOT.on(*qreg), Z.on(qreg[1])])

    # Invert circuit
    inverse_circuit = cirq.inverse(circuit)
    inverse_repr = inverse_circuit.__repr__()
    inverse_qasm = inverse_circuit._to_qasm_output().__str__()

    # Expected circuit after simplification
    expected_inv = Circuit([Z.on(qreg[1]), CNOT.on(*qreg), H.on(qreg[0])])
    expected_repr = expected_inv.__repr__()
    expected_qasm = expected_inv._to_qasm_output().__str__()

    # Check inverse_circuit is logically equivalent to expected_inverse
    # but they have a different representation
    assert inverse_circuit == expected_inv
    assert inverse_repr != expected_repr
    assert inverse_qasm != expected_qasm

    # Simplify the circuit
    _simplify_circuit_exponents(inverse_circuit)

    # Check inverse_circuit has the expected simplified representation
    simplified_repr = inverse_circuit.__repr__()
    simplified_qasm = inverse_circuit._to_qasm_output().__str__()
    assert inverse_circuit == expected_inv
    assert simplified_repr == expected_repr
    assert simplified_qasm == expected_qasm


def test_simplify_circuit_exponents_with_non_self_inverse_gates():
    qreg = LineQubit.range(2)
    # Make a circuit with gates which are not self-inverse
    circuit = Circuit([S.on(qreg[0]), T.on(qreg[1])])

    inverse_circuit = cirq.inverse(circuit)
    inverse_repr = inverse_circuit.__repr__()
    inverse_qasm = inverse_circuit._to_qasm_output().__str__()

    # Simplify the circuit (it should not change this circuit)
    _simplify_circuit_exponents(inverse_circuit)

    # Check inverse_circuit did not change
    simplified_repr = inverse_circuit.__repr__()
    simplified_qasm = inverse_circuit._to_qasm_output().__str__()
    assert simplified_repr == inverse_repr
    assert simplified_qasm == inverse_qasm


def test_are_close_dict():
    """Tests the _are_close_dict function."""
    dict1 = {"a": 1, "b": 0.0}
    dict2 = {"a": 1, "b": 0.0 + 1.0e-10}
    assert _are_close_dict(dict1, dict2)
    assert _are_close_dict(dict2, dict1)
    dict2 = {"b": 0.0 + 1.0e-10, "a": 1}
    assert _are_close_dict(dict1, dict2)
    assert _are_close_dict(dict2, dict1)
    dict2 = {"a": 1, "b": 1.0}
    assert not _are_close_dict(dict1, dict2)
    assert not _are_close_dict(dict2, dict1)
    dict2 = {"b": 1, "a": 0.0}
    assert not _are_close_dict(dict1, dict2)
    assert not _are_close_dict(dict2, dict1)
    dict2 = {"a": 1, "b": 0.0, "c": 1}
    assert not _are_close_dict(dict1, dict2)
    assert not _are_close_dict(dict2, dict1)


def test_max_ent_state_circuit():
    """Tests 1-qubit and 2-qubit maximally entangled states are generated."""
    two_state = np.array([1, 0, 0, 1]) / np.sqrt(2)
    four_state = np.array(3 * [1, 0, 0, 0, 0] + [1]) / 2.0
    assert np.allclose(
        _max_ent_state_circuit(2).final_state_vector(), two_state
    )
    assert np.allclose(
        _max_ent_state_circuit(4).final_state_vector(), four_state
    )


def test_circuit_to_choi_and_operation_to_choi():
    """Tests the Choi matrix of a depolarizing channel is recovered."""
    # Define first the expected result
    base_noise = 0.01
    max_ent_state = np.array([1, 0, 0, 1]) / np.sqrt(2)
    identity_part = np.outer(max_ent_state, max_ent_state)
    mixed_part = np.eye(4) / 4.0
    epsilon = base_noise * 4.0 / 3.0
    choi = (1.0 - epsilon) * identity_part + epsilon * mixed_part
    # Choi matrix of the double application of a depolarizing channel
    choi_twice = sum(
        [
            ((1.0 - epsilon) ** 2 * identity_part),
            (2 * epsilon - epsilon ** 2) * mixed_part,
        ]
    )

    # Evaluate the Choi matrix of one or two depolarizing channels
    q = LineQubit(0)
    noisy_operation = depolarize(base_noise).on(q)
    noisy_sequence = [noisy_operation, noisy_operation]
    assert np.allclose(choi, _operation_to_choi(noisy_operation))
    noisy_circuit = Circuit(noisy_operation)
    noisy_circuit_twice = Circuit(noisy_sequence)
    assert np.allclose(choi, _circuit_to_choi(noisy_circuit))
    print(noisy_circuit_twice)
    print(_circuit_to_choi(noisy_circuit_twice))
    print(choi_twice)
    assert np.allclose(choi_twice, _circuit_to_choi(noisy_circuit_twice))
