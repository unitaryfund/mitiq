# Copyright (C) 2022 Unitary Fund
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Functions for creating circuits of the form used in quantum
volume experiments as defined in https://arxiv.org/abs/1811.12926.

Useful overview of quantum volume experiments:
https://pennylane.ai/qml/demos/quantum_volume.html

Cirq implementation of quantum volume circuits:
cirq-core/cirq/contrib/quantum_volume/quantum_volume.py

The following code draws heavily on the Cirq implementation.
"""

from typing import Optional, List, Tuple


import numpy as np
import cirq
from mitiq import QPROGRAM
from mitiq.interface import convert_from_mitiq
from mitiq.rem.measurement_result import Bitstring # List[int]


def generate_volume_circuit(
    num_qubits: int,
    return_type: Optional[str] = None,
) -> Tuple[QPROGRAM, List[Bitstring]:
    """Returns a quantum volume circuit ie a circuit that 
    TODO: finish description

    Args:
        num_qubits: The number of qubits in the circuit (and, since
        it is a square circuit, num_qubits is also the circuit depth).
        return_type: String which specifies the type of the returned
            circuits. See the keys of ``mitiq.SUPPORTED_PROGRAM_TYPES``
            for options. If ``None``, the returned circuits have type
            ``cirq.Circuit``.

    Returns:
        A quantum volume circuit acting on ``num_qubits`` qubits.
        A list of the heavy bitstrings for the returned circuit.
    """

    if num_qubits <= 0:
        raise ValueError(
            "Cannot prepare a volume circuit with {} qubits", num_qubits
        )

    qubits = cirq.LineQubit.range(num_qubits)
    circuit = cirq.Circuit()
    random_state = np.random_state

    # For each circuit layer
    for _ in range (num_qubits):
        # Generate uniformly random permutation Pj of [0...n-1]
        perm = random_state.permutation(num_qubits)

        # For each consecutive pair in Pj, generate Haar random SU(4)
        # Decompose each SU(4) into CNOT + SU(2) and add to Ci
        for k in range(0, num_qubits - 1, 2):
            permuted_indices = [int(perm[k]), int(perm[k + 1])]
            special_unitary = cirq.testing.random_special_unitary(4, random_state=random_state)

            # Convert the decomposed unitary to Cirq operations and add them to
            # the circuit.
            circuit.append(
                cirq.MatrixGate(special_unitary).on(
                    qubits[permuted_indices[0]], qubits[permuted_indices[1]]
                )
            )

    # Don't measure all of the qubits at the end of the circuit because we will
    # need to classically simulate it to compute its heavy set.
    heavy_bitstrings = compute_heavy_set(circuit) 

    return_type = "cirq" if not return_type else return_type
    circuit = convert_from_mitiq(circuit, return_type) 

    return circuit, heavy_bitstrings 


def compute_heavy_set(circuit: cirq.Circuit) -> Bitstring:
    """Classically compute the heavy set of the given circuit.

    The heavy set is defined as the output bit-strings that have a greater than
    median probability of being generated.

    Args:
        circuit: The circuit to classically simulate.

    Returns:
        A list containing all of the heavy bit-string results.
    """
    # Classically compute the probabilities of each output bit-string through
    # simulation.
    simulator = cirq.Simulator()
    results = cast(cirq.StateVectorTrialResult, simulator.simulate(program=circuit))

    # Compute the median probability of the output bit-strings. Note that heavy
    # output is defined in terms of probabilities, where our wave function is in
    # terms of amplitudes. We convert it by using the Born rule: squaring each
    # amplitude and taking their absolute value
    median = np.median(np.abs(results.state_vector() ** 2))

    # The output wave function is a vector from the result value (big-endian) to
    # the probability of that bit-string. Return all of the bit-string
    # values that have a probability greater than the median.
    return [idx for idx, amp in enumerate(results.state_vector()) if np.abs(amp**2) > median]
