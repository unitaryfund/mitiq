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

from typing import Optional, List, Tuple, cast

from numpy import random

from cirq.contrib.quantum_volume import (
    generate_model_circuit,
    compute_heavy_set,
)
from cirq.value import big_endian_int_to_bits

from mitiq import QPROGRAM
from mitiq.interface import convert_from_mitiq
from mitiq.rem.measurement_result import Bitstring # List[int]


def generate_volume_circuit(
    num_qubits: int,
    depth: int,
    seed: Optional[int] = None,
    return_type: Optional[str] = None,
) -> Tuple[QPROGRAM, List[Bitstring]]:
    """Generates a volume circuit with the given number of qubits and depth.

    The generated circuit consists of `depth` layers of random qubit
    permutations followed by random two-qubit gates that are sampled from the
    Haar measure on SU(4).

    Args:
        num_qubits: The number of qubits in the generated circuit. 
        depth: The number of qubits in the generated circuit.
        seed: Seed for generating random circuit.
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
    
    random_state = random.RandomState(seed)

    circuit = generate_model_circuit(num_qubits, depth, random_state=random_state)
    heavy_vals = compute_heavy_set(circuit)

    # Convert base-10 ints to Bitstrings.
    heavy_bitstrings = [big_endian_int_to_bits(val, bit_count=num_qubits)
            for val in heavy_vals]

    return_type = "cirq" if not return_type else return_type
    circuit = convert_from_mitiq(circuit, return_type) 

    return circuit, heavy_bitstrings


