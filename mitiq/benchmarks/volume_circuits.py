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

from cirq import decompose as cirq_decompose
from cirq import Circuit

from cirq.contrib.quantum_volume import (
    generate_model_circuit,
    compute_heavy_set,
)

from mitiq import QPROGRAM
from mitiq.interface import convert_from_mitiq
from mitiq.rem.measurement_result import Bitstring # List[int]


def generate_volume_circuit(
    num_qubits: int,
    depth: int,
    return_type: Optional[str] = None,
    decompose: bool = False,
) -> Tuple[QPROGRAM, List[Bitstring]]:
    """Generates a volume circuit with the given number of qubits and depth.

    The generated circuit consists of `depth` layers of random qubit
    permutations followed by random two-qubit gates that are sampled from the
    Haar measure on SU(4).

    Args:
        num_qubits: The number of qubits in the generated circuit. 
        depth: The number of qubits in the generated circuit.
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
    
    circuit = generate_model_circuit(num_qubits, depth)
    heavy_set = compute_heavy_set(circuit)
    if decompose:
        circuit = Circuit(cirq_decompose(circuit))

    return_type = "cirq" if not return_type else return_type
    circuit = convert_from_mitiq(circuit, return_type) 

    return circuit, heavy_set 
