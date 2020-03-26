"""Unit test for utility functions."""

import pytest

import cirq

from mitiq.utils import _equal, random_circuit


def test_random_circuit_simple():
    n = 100
    circuit = random_circuit(depth=n)
    assert len(list(circuit.all_operations())) == n


@pytest.mark.parametrize(["require_qubit_equality"], [[True], [False]])
def test_circuit_equality_identical_qubits(require_qubit_equality):
    qreg = cirq.NamedQubit.range(5, prefix="q_")
    circA = cirq.Circuit(cirq.ops.H.on_each(*qreg))
    circB = cirq.Circuit(cirq.ops.H.on_each(*qreg))
    assert circA is not circB
    assert _equal(circA, circB, require_qubit_equality=require_qubit_equality)


@pytest.mark.parametrize(["require_qubit_equality"], [[True], [False]])
def test_circuit_equality_nonidentical_but_equal_qubits(require_qubit_equality,):
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
