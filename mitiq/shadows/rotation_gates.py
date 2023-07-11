from typing import List
import cirq
import numpy as np

# Functions to implement random Pauli measurements.


def generate_random_pauli_strings(
    num_qubits: int, num_strings: int
) -> List[str]:
    """Generate a list of random Pauli strings.

    Args:
        num_qubits: The number of qubits in the Pauli strings.
        num_strings: The number of Pauli strings to generate.

    Returns:
        A list of random Pauli strings.
    """

    unitary_ensemble = ["X", "Y", "Z"]
    paulis = np.random.choice(unitary_ensemble, (num_strings, num_qubits))
    return ["".join(pauli) for pauli in paulis]


def get_rotated_circuits(
    circuit: cirq.Circuit, pauli_strings: List[str]
) -> List[cirq.Circuit]:
    """Returns a list of circuits that are identical to the input circuit,
    except that each one has single-qubit Clifford gates followed by
    measurement gates that are designed to measure the input
    Pauli strings in the Z basis.

    Args:
        circuit: The circuit to measure.
        pauli_strings: The Pauli strings to measure in each output circuit.

    Returns:
        The list of circuits with rotation and measurement gates appended.
    """
    qubits = list(circuit.all_qubits())
    num_qubits = len(qubits)
    rotated_circuits = []
    for pauli_string in pauli_strings:
        assert len(pauli_string) == num_qubits, (
            f"Pauli string must be same length as number of qubits,"
            f" got {len(pauli_string)} and {num_qubits}"
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
