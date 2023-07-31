# Copyright (C) Unitary Fund
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

"""Quantum processing functions for classical shadows."""
from typing import Tuple, Callable, Any, List, Optional
import cirq
import numpy as np

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

from mitiq import MeasurementResult


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

    # Sample random Pauli operators uniformly from ("X", "Y", "Z")
    unitary_ensemble = ["X", "Y", "Z"]
    paulis = np.random.choice(unitary_ensemble, (num_strings, num_qubits))
    return ["".join(pauli) for pauli in paulis]


def get_rotated_circuits(
    circuit: cirq.Circuit,
    pauli_strings: List[str],
    add_measurements: bool = True,
    qubits: Optional[List[cirq.Qid]] = None,
) -> List[cirq.Circuit]:
    """Returns a list of circuits that are identical to the input circuit,
    except that each one has single-qubit Clifford gates followed by
    measurement gates that are designed to measure the input
    Pauli strings in the Z basis.

    Args:
        circuit: The circuit to measure.
        pauli_strings: The Pauli strings to measure in each output circuit.
        add_measurements: Whether to add measurement gates to the circuit.
        qubits: The qubits to measure. If None, all qubits in the circuit
    Returns:
         The list of circuits with rotation and measurement gates appended.
    """
    qubits = sorted(list(circuit.all_qubits())) if qubits is None else qubits
    rotated_circuits = []
    for pauli_string in pauli_strings:
        rotated_circuit = circuit.copy()
        for qubit, pauli in zip(qubits, pauli_string):
            # Pauli X measurement is equivalent to H plus a Z measurement
            if pauli == "X":
                rotated_circuit.append(cirq.H(qubit))
            # Pauli X measurement is equivalent to S^-1*H plus a Z measurement
            elif pauli == "Y":
                rotated_circuit.append(cirq.S(qubit) ** -1)
                rotated_circuit.append(cirq.H(qubit))
            # Pauli Z measurement
            else:
                assert (
                    pauli == "Z"
                ), f"Pauli must be X, Y, Z. Got {pauli} instead."
        if add_measurements:
            rotated_circuit.append(cirq.measure(*qubits))
        rotated_circuits.append(rotated_circuit)
    return rotated_circuits


def random_pauli_measurement(
    circuit: cirq.Circuit,
    n_total_measurements: int,
    executor: Callable[[cirq.Circuit], MeasurementResult],
    qubits: Optional[List[cirq.Qid]] = None,
) -> Tuple[List[str], List[str]]:
    r"""
    Given a circuit, perform random Pauli measurements on the circuit and
    return outcomes. The outcomes are represented as a tuple of two lists of
    strings.

    Args:
        circuit: Cirq circuit.
        n_total_measurements: number of snapshots.
        executor: A callable which runs a circuit and returns a single
            bitstring.
        qubits: The qubits to measure. If None, all qubits in the circuit

    Warning:
        The ``executor`` must return a ``MeasurementResult``
        for a single shot (i.e. a single bitstring).

    Returns:
        Tuple of two list of string of length equals to`n_total_meausrements`,
        where each element in the list is a string, with the following format:
        strs in the 1st list: Circuit qubits computational basis
            e.g. :math:`"01..":=|0\rangle|1\rangle..`.
        strs in the 1st list: The local Pauli measurement performed on each
            qubit. e.g."XY.." means perform local X-basis measurement on the
            1st qubit, local Y-basis measurement the 2ed qubit in the circuit.
    """
    qubits = sorted(list(circuit.all_qubits())) if qubits is None else qubits
    num_qubits = len(qubits)
    pauli_strings = generate_random_pauli_strings(
        num_qubits, n_total_measurements
    )

    # Rotate and attach measurement gates to the circuit
    rotated_circuits = get_rotated_circuits(
        circuit=circuit,
        pauli_strings=pauli_strings,
        add_measurements=True,
        qubits=qubits,
    )

    if tqdm is not None:
        rotated_circuits = tqdm(
            rotated_circuits,
            desc="Measurement",
            leave=False,
        )
    results = [
        executor(rotated_circuit) for rotated_circuit in rotated_circuits
    ]

    shadow_outcomes = []
    for result in results:
        bitstring = list(result.get_counts().keys())[0]
        if len(result.get_counts().keys()) > 1:
            raise ValueError(
                "The `executor` must return a `MeasurementResult` "
                "for a single shot"
            )
        shadow_outcomes.append(bitstring)

    return shadow_outcomes, pauli_strings
