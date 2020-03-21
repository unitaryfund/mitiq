"""Utility functions."""

import random

import cirq


def random_circuit(depth: int, **kwargs) -> cirq.Circuit:
    """Returns a random single-qubit circuit with Pauli gates."""
    if "seed" in kwargs.keys():
        random.seed(kwargs.get("seed"))

    qubit = cirq.GridQubit(0, 0)
    gates = [cirq.ops.X, cirq.ops.Y, cirq.ops.Z]
    circuit = cirq.Circuit([random.choice(gates).on(qubit) for _ in range(depth)])

    return circuit


def _equal(circuit_one: cirq.Circuit, circuit_two: cirq.Circuit) -> bool:
    """Returns True if circuits are equal.

    Args:
        circuit_one: Input circuit to compare to circuit_two.
        circuit_two: Input circuit to compare to circuit_one.
    """
    return cirq.CircuitDag.from_circuit(circuit_one) == cirq.CircuitDag.from_circuit(circuit_two)
