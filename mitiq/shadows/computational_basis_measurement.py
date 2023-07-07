import cirq
import numpy as np
from typing import List, Tuple, Union, Dict, Any
import cirq
from mitiq import Executor, Observable, PauliString, QPROGRAM
from mitiq.typing import QuantumResult, MeasurementResult
# generate N random Pauli strings for given number of qubits
def generate_random_pauli_strings(num_qubits: int, num_strings: int) -> List[str]:
    # return "XXXYZYZX" where len string == num_qubits
    
    # Sample random Pauli operators uniformly from X, Y, Z
    unitary_ensemble = ["X", "Y", "Z"]
    paulis = np.random.choice(unitary_ensemble, (num_strings, num_qubits))
    return ["".join(pauli) for pauli in paulis]

# attach random rotate gates to N copies of the circuit
def get_rotated_circuits(
    circuit: cirq.Circuit, pauli_strings: List[str]
) -> List[cirq.Circuit]:
    """Returns a list of circuits that are identical to the given circuit, except that each one has a different Pauli gate applied to each qubit, followed by a measurement.

    Args:
        circuit: The circuit to measure.
        pauli_strings: The Pauli strings to apply to each qubit, in order, before measuring.

    Returns:
        A list of circuits, one for each Pauli string.
    """
    qubits = list(circuit.all_qubits())
    num_qubits = len(qubits)
    rotated_circuits = []
    for pauli_string in pauli_strings:
        assert len(pauli_string) == num_qubits, f"Pauli string must be same length as number of qubits, got {len(pauli_string)} and {num_qubits}"
        rotated_circuit = circuit.copy()
        for i, pauli in enumerate(pauli_string):
            qubit = qubits[i]
            if pauli == "X":
                rotated_circuit.append(cirq.H(qubit))
            elif pauli == "Y":
                rotated_circuit.append(cirq.S(qubit) ** -1)
                rotated_circuit.append(cirq.H(qubit))
            else:
                assert pauli == "Z" or pauli == "I", f"Pauli must be X, Y, Z or I. Got {pauli} instead."
            if pauli != "I":
                rotated_circuit.append(cirq.measure(qubit))
        rotated_circuits.append(rotated_circuit)
    return rotated_circuits

# Stage 1 of Classical Shadows: Measurements
def shadow_measure_with_executor(
        circuit: cirq.Circuit,
        executor: Executor, 
        n_total_measurements: int, ) -> Tuple[np.ndarray, np.ndarray]:
    """
    
    Given a circuit, calculate the classical shadow of the circuit of aa bit
      string, and the index of Pauli matrices that were measured.

    Args: circuit (cirq.Circuit): Cirq circuit.
          n_total_measurements (int): number of snapshots.

    Returns: outcomes (array): Tuple of two numpy arrays. The first array contains measurement outcomes (-1, 1)
        while the second array contains the index for the sampled Pauli's (0,1,2=X,Y,Z).
        Each row of the arrays corresponds to a distinct snapshot or sample while each
        column corresponds to a different qubit.
    """
 
    # Generate random Pauli unitaries
    qubits = list(circuit.all_qubits())
    num_qubits = len(qubits)
    pauli_strings = generate_random_pauli_strings(num_qubits, n_total_measurements)
    # Attach measurement gates to the circuit
    rotated_circuits = get_rotated_circuits(circuit, pauli_strings)
    
    # Run the circuits to collect the outcomes
    results = executor.evaluate(rotated_circuits)

    # Transform the outcomes into a numpy array.
    shadow_outcomes = []
    for result in results:
        bitstring = list(result.get_counts().keys())[0]
        outcome = [ 1-int(i)*2 for i in bitstring]
        shadow_outcomes.append(outcome)
    # Combine the computational basis outcomes $$|\mathbf{b}\rangle$$ and the unitaries sampled from $$CL_2^{\otimes n}$$.
    shadow_outcomes = np.array(shadow_outcomes, dtype=int)
    assert shadow_outcomes.shape == (n_total_measurements, num_qubits), f"shape is {shadow_outcomes.shape}"
    pauli_strings = np.array(pauli_strings, dtype=str)
    return shadow_outcomes, pauli_strings