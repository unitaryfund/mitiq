"""Utility functions."""

from typing import Optional

import random

import cirq


def random_circuit(depth: int, seed: Optional[int] = None) -> cirq.Circuit:
    """Returns a random single-qubit circuit with Pauli gates.

    Parameters
    ----------
        depth: Number of gates in the circuit.
        seed: Seed for the random number generator.
    """
    if seed:
        random.seed(seed)

    qubit = cirq.GridQubit(0, 0)
    gates = [cirq.ops.X, cirq.ops.Y, cirq.ops.Z]
    circuit = cirq.Circuit(
        [random.choice(gates).on(qubit) for _ in range(depth)]
    )

    return circuit


def _equal(
        circuit_one: cirq.Circuit,
        circuit_two: cirq.Circuit,
        require_qubit_equality: bool = False
) -> bool:
    """Returns True if the circuits are equal, else False.

    Parameters
    ----------
        circuit_one: Input circuit to compare to circuit_two.
        circuit_two: Input circuit to compare to circuit_one.
        require_qubit_equality: Requires that the qubits be equal
            in the two circuits.

            E.g., if set(circuit_one.all_qubits()) = {LineQubit(0)},
            then set(circuit_two_all_qubits()) must be {LineQubit(0)},
            else the two are not equal.

            If True, the qubits of both circuits must have
            a well-defined ordering.
    """
    if circuit_one is circuit_two:
        return True

    if not require_qubit_equality:
        # Transform the qubits of circuit one to those of circuit two
        qubit_map = dict(zip(
            sorted(circuit_one.all_qubits()), sorted(circuit_two.all_qubits())
        ))
        circuit_one = circuit_one.transform_qubits(lambda q: qubit_map[q])

    return (cirq.CircuitDag.from_circuit(circuit_one)
            ==
            cirq.CircuitDag.from_circuit(circuit_two))
