# Copyright (C) Unitary Fund
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

"""Quantum processing functions for classical shadows."""
from typing import Tuple, Callable, Any, List

import cirq
import numpy as np
from numpy.typing import NDArray

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
    bitflip_ratio: float = 0.00,
) -> List[cirq.Circuit]:
    """Returns a list of circuits that are identical to the input circuit,
    except that each one has single-qubit Clifford gates followed by
    measurement gates that are designed to measure the input
    Pauli strings in the Z basis.

    Args:
        circuit: The circuit to measure.
        pauli_strings: The Pauli strings to measure in each output circuit.
        add_measurements: Whether to add measurement gates to the circuit.

    Returns:
         The list of circuits with rotation and measurement gates appended.
    """
    qubits = sorted(list(circuit.all_qubits()))
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
        if bitflip_ratio > 0.0:
            rotated_circuit.append(
                cirq.bit_flip(bitflip_ratio).on_each(*qubits)
            )
        if add_measurements:
            rotated_circuit.append(cirq.measure(*qubits))
        rotated_circuits.append(rotated_circuit)
    return rotated_circuits


def random_pauli_measurement(
    circuit: cirq.Circuit,
    n_total_measurements: int,
    executor: Callable[[cirq.Circuit], MeasurementResult],
) -> Tuple[NDArray[Any], NDArray[Any]]:
    r"""
    Given a circuit, perform random Pauli measurements on the circuit and
    return outcomes. The outcomes are represented as a string where a
    z-basis measurement outcome of :math:`|0\rangle` corresponds to 1, and
    :math:`|1\rangle` corresponds to -1.

    Args:
        circuit: Cirq circuit.
        n_total_measurements: number of snapshots.
        executor: A callable which runs a circuit and returns a single
            bitstring.

    Warning:
        The ``executor`` must return a ``MeasurementResult``
        for a single shot (i.e. a single bitstring).

    Returns:
        Tuple of two numpy arrays. The first array contains
        measurement outcomes (-1, 1) while the second array contains the
        indices for the sampled Pauli's ("X", "Y", "Z").
        This implies that local Clifford rotations plus z-basis measurements
        are effectively equivalent to random Pauli measurements.
        Each row of the arrays corresponds to a distinct snapshot or sample,
        while each column corresponds to measurement outcomes
        and random Pauli measurement on a different qubit.
    """

    num_qubits = len(circuit.all_qubits())
    pauli_strings = generate_random_pauli_strings(
        num_qubits, n_total_measurements
    )

    # Rotate and attach measurement gates to the circuit
    rotated_circuits = get_rotated_circuits(
        circuit,
        pauli_strings,
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

    # Transform the outcomes into a numpy array 0 -> 1, 1 -> -1.
    shadow_outcomes = []
    for result in results:
        bitstring = list(result.get_counts().keys())[0]
        if len(result.get_counts().keys()) > 1:
            raise ValueError(
                "The `executor` must return a `MeasurementResult` "
                "for a single shot"
            )

        outcome = [1 - int(i) * 2 for i in bitstring]
        shadow_outcomes.append(outcome)

    shadow_outcomes_np = np.asarray(shadow_outcomes, dtype=int)
    pauli_strings_np = np.asarray(pauli_strings, dtype=str)

    return shadow_outcomes_np, pauli_strings_np
