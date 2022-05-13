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

Useful overview of quantum volume experiments here:
https://pennylane.ai/qml/demos/quantum_volume.html

Cirq implementation of quantum volume circuits here:
cirq-core/cirq/contrib/quantum_volume/quantum_volume.py
"""

from typing import Optional, List, Tuple

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
        n_qubits: The number of qubits in the circuit.
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

    #TODO

    return circuit, heavy_bitstrings 



