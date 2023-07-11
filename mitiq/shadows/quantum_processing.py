from typing import Tuple, Callable, Optional, Dict, Any, List, Union

import cirq
import numpy as np
from qiskit_aer import Aer

from mitiq.interface.mitiq_cirq.cirq_utils import (
    sample_bitstrings as cirq_sample_bitstrings,
)
from mitiq.interface.mitiq_qiskit.conversions import to_qiskit
from mitiq.interface.mitiq_qiskit.qiskit_utils import (
    sample_bitstrings as qiskit_sample_bitstrings,
)
from tqdm.auto import tqdm

# generate N random Pauli strings for given number of qubits


def generate_random_pauli_strings(
    num_qubits: int, num_strings: int
) -> List[str]:
    """Generate a list of random Pauli strings.

    Args:
        num_qubits: The number of qubits in the Pauli strings.
        num_strings: The number of Pauli strings to generate.

    Returns:
        A list of number of `num_strings` random Pauli strings
          of length `num_qubits`.
    """

    # Sample random Pauli operators uniformly from X, Y, Z
    unitary_ensemble = ["X", "Y", "Z"]
    paulis = np.random.choice(unitary_ensemble, (num_strings, num_qubits))
    return ["".join(pauli) for pauli in paulis]


# attach random rotate gates to N copies of the circuit
def get_rotated_circuits(
    circuit: cirq.Circuit,
    pauli_strings: List[str],
    add_measurements: bool = True,
) -> List[cirq.Circuit]:
    """Returns a list of circuits that are identical to the given circuit,
    except that each one has a different Pauli gate applied to each qubit,
    followed by a measurement.

    Args:
        circuit: The circuit to measure.
        pauli_strings: The Pauli strings to apply to each qubit, in order,
            before measuring.
        add_measurements: Whether to add measurement gates to the circuit.

    Returns:
        A list of circuits, one for each Pauli string.
    """
    qubits = list(circuit.all_qubits())
    num_qubits = len(qubits)
    rotated_circuits = []
    for pauli_string in pauli_strings:
        assert (
            len(pauli_string) == num_qubits
        ), f"Pauli string must be same length as number of qubits, got {len(pauli_string)} and {num_qubits}"
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
        if add_measurements:
            rotated_circuit = add_measurement_gates(rotated_circuit)
        rotated_circuits.append(rotated_circuit)
    return rotated_circuits


def add_measurement_gates(circuit: cirq.Circuit) -> cirq.Circuit:
    """Add measurement gates to a circuit.

    Args:
        circuit: The circuit to measure.

    Returns:
        The circuit with measurement gates appended.
    """
    qubits = list(circuit.all_qubits())

    # append measurement gates
    circuit.append(cirq.measure(*qubits))
    return circuit


# Stage 1 of Classical Shadows: Measurements
def get_z_basis_measurement(
    circuit: cirq.Circuit,
    n_total_measurements: int,
    sampling_function: Optional[Union[str, Callable]] = "cirq",
    sampling_function_config: Optional[Dict[str, Any]] = {},
) -> Tuple[np.ndarray, np.ndarray]:
    r"""Given a circuit, perform z-basis measurements on the circuit and return
    the outcomes in terms of a string, which represents for z-basis measurement
    outcomes $$1:=\{1,0\}$$, $$-1:=\{0,1\}$$.

    Args: circuit (cirq.Circuit): Cirq circuit.
              n_total_measurements (int): number of snapshots.
                sampling_function (Optional[Union[str, Callable]]): Sampling function to use. If None, then
                    the default sampling function for the backend is used. If a string, then the string
                    is used to look up a sampling function in the backend's sampling function registry.
                    If a callable, then the callable is used as the sampling function. Defaults to None.
                sampling_function_config (Optional[Dict[str, Any]]): Configuration for the sampling function.
                    Defaults to {}. If sampling_function is None, then this argument is ignored.

    Returns: outcomes (array): Tuple of two numpy arrays. The first array contains measurement outcomes (-1, 1)
                    while the second array contains the index for the sampled Pauli's (0,1,2=X,Y,Z).
                    Each row of the arrays corresponds to a distinct snapshot or sample while each
                    column corresponds to a different qubit.
    """

    # Generate random Pauli unitaries
    qubits = list(circuit.all_qubits())
    num_qubits = len(qubits)
    pauli_strings = generate_random_pauli_strings(
        num_qubits, n_total_measurements
    )
    # Attach measurement gates to the circuit
    rotated_circuits = tqdm(
        get_rotated_circuits(
            circuit,
            pauli_strings,
        ),
        desc="Measurement",
        leave=False,
    )

    if isinstance(sampling_function, str):
        if sampling_function == "cirq":
            # Run the circuits to collect the outcomes for cirq)
            results = [
                cirq_sample_bitstrings(
                    rotated_circuit,
                    noise_level=(0,),
                    shots=1,
                    sampler=cirq.Simulator(),
                )
                for rotated_circuit in rotated_circuits
            ]
        elif sampling_function == "qiskit":
            # Run the circuits to collect the outcomes for cirq
            results = [
                qiskit_sample_bitstrings(
                    to_qiskit(rotated_circuit),
                    noise_model=None,
                    backend=Aer.get_backend("aer_simulator"),
                    shots=1,
                    measure_all=False,
                )
                for rotated_circuit in rotated_circuits
            ]
    else:
        assert isinstance(
            sampling_function, Callable
        ), "Please define your own sample_bitstrings function"
        results = [
            sampling_function(rotated_circuit, **sampling_function_config)
            for rotated_circuit in rotated_circuits
        ]

    # Transform the outcomes into a numpy array.
    shadow_outcomes = []
    for result in results:
        bitstring = list(result.get_counts().keys())[0]
        outcome = [1 - int(i) * 2 for i in bitstring]
        shadow_outcomes.append(outcome)
    # Combine the computational basis outcomes $$|\mathbf{b}\rangle$$ and the unitaries sampled from $$CL_2^{\otimes n}$$.
    shadow_outcomes = np.array(shadow_outcomes, dtype=int)
    assert shadow_outcomes.shape == (
        n_total_measurements,
        num_qubits,
    ), f"shape is {shadow_outcomes.shape}"
    pauli_strings = np.array(pauli_strings, dtype=str)
    return shadow_outcomes, pauli_strings
