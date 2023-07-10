from typing import List

import cirq
import numpy as np


# generate N random Pauli strings for given number of qubits
def generate_random_pauli_strings(
    num_qubits: int, num_strings: int
) -> List[str]:
    # return "XXXYZYZX" where len string == num_qubits

    # Sample random Pauli operators uniformly from X, Y, Z
    unitary_ensemble = ["X", "Y", "Z"]
    paulis = np.random.choice(unitary_ensemble, (num_strings, num_qubits))
    return ["".join(pauli) for pauli in paulis]


# attach random rotate gates to N copies of the circuit
def get_rotated_circuits(
    circuit: cirq.Circuit, pauli_strings: List[str]
) -> List[cirq.Circuit]:
    """Returns a list of circuits that are identical to the given circuit,
    except that each one has a different Pauli gate applied to each qubit,
    followed by a measurement.

    Args:
        circuit: The circuit to measure.
        pauli_strings: The Pauli strings to apply to each qubit, in order,
        before measuring.

    Returns:
        A list of circuits, one for each Pauli string.
    """
    qubits = list(circuit.all_qubits())
    num_qubits = len(qubits)
    rotated_circuits = []
    for pauli_string in pauli_strings:
        assert len(pauli_string) == num_qubits, (
            f"Pauli string must be same length as number of qubits, "
            f"got {len(pauli_string)} and {num_qubits}"
        )
        rotated_circuit = circuit.copy()
        for i, pauli in enumerate(pauli_string):
            qubit = qubits[i]
            if pauli == "X":
                rotated_circuit.append(cirq.H(qubit))
            elif pauli == "Y":
                rotated_circuit.append(cirq.S(qubit) ** -1)
                rotated_circuit.append(cirq.H(qubit))
            else:
                assert (
                    pauli == "Z" or pauli == "I"
                ), f"Pauli must be X, Y, Z or I. Got {pauli} instead."
            if pauli != "I":
                rotated_circuit.append(cirq.measure(qubit))
        rotated_circuits.append(rotated_circuit)
    return rotated_circuits
