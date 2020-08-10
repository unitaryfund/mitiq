"""Tests for utility functions."""

from copy import deepcopy
import pytest

import cirq

from mitiq.utils import _are_close_dict, _equal


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
