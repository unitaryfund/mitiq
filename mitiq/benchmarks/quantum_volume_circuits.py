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
"""

from typing import Optional, List, Tuple

from numpy import random

from cirq import decompose as cirq_decompose
from cirq.circuits import Circuit
from cirq.contrib.quantum_volume import (
    generate_model_circuit,
    compute_heavy_set,
)
from cirq.value import big_endian_int_to_bits


from mitiq import QPROGRAM
from mitiq._typing import Bitstring
from mitiq.interface import convert_from_mitiq


def generate_quantum_volume_circuit(
    num_qubits: int,
    depth: int,
    decompose: bool = False,
    seed: Optional[int] = None,
    return_type: Optional[str] = None,
) -> Tuple[QPROGRAM, List[Bitstring]]:
    """Generate a quantum volume circuit with the given number of qubits and
    depth.

    The generated circuit consists of `depth` layers of random qubit
    permutations followed by random two-qubit gates that are sampled from the
    Haar measure on SU(4).

    Args:
        num_qubits: The number of qubits in the generated circuit.
        depth: The number of qubits in the generated circuit.
        decompose: Recursively decomposes the randomly sampled (numerical)
            unitary matrix gates into simpler gates.
        seed: Seed for generating random circuit.
        return_type: String which specifies the type of the returned
            circuits. See the keys of ``mitiq.SUPPORTED_PROGRAM_TYPES``
            for options. If ``None``, the returned circuits have type
            ``cirq.Circuit``.

    Returns:
        A quantum volume circuit acting on ``num_qubits`` qubits.
        A list of the heavy bitstrings for the returned circuit.
    """
    random_state = random.RandomState(seed)
    circuit = generate_model_circuit(
        num_qubits, depth, random_state=random_state
    )
    heavy_bitstrings = compute_heavy_bitstrings(circuit, num_qubits)

    if decompose:
        # Decompose random unitary gates into simpler gates.
        circuit = Circuit(cirq_decompose(circuit))

    return_type = "cirq" if not return_type else return_type
    return convert_from_mitiq(circuit, return_type), heavy_bitstrings


def compute_heavy_bitstrings(
    circuit: Circuit,
    num_qubits: int,
) -> List[Bitstring]:
    """Classically compute the heavy bitstrings of the provided circuit.

    The heavy bitstrings are defined as the output bit-strings that have a
    greater than median probability of being generated.

    Args:
        circuit: The circuit to classically simulate.

    Returns:
        A list containing the heavy bitstrings.
    """
    heavy_vals = compute_heavy_set(circuit)
    # Convert base-10 ints to Bitstrings.
    heavy_bitstrings = [
        big_endian_int_to_bits(val, bit_count=num_qubits) for val in heavy_vals
    ]
    return heavy_bitstrings
