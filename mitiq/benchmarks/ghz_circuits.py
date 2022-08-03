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

"""Functions for creating GHZ circuits for benchmarking purposes."""
from typing import Optional

import cirq
from mitiq import QPROGRAM
from mitiq.interface import convert_from_mitiq


def generate_ghz_circuit(
    n_qubits: int,
    return_type: Optional[str] = None,
) -> QPROGRAM:
    """Returns a GHZ circuit ie a circuit that prepares an ``n_qubits``
    GHZ state.

    Args:
        n_qubits: The number of qubits in the circuit.
        return_type: String which specifies the type of the returned
            circuits. See the keys of ``mitiq.SUPPORTED_PROGRAM_TYPES``
            for options. If ``None``, the returned circuits have type
            ``cirq.Circuit``.

    Returns:
        A GHZ circuit acting on ``n_qubits`` qubits.
    """
    if n_qubits <= 0:
        raise ValueError(
            "Cannot prepare a GHZ circuit with {} qubits", n_qubits
        )

    qubits = cirq.LineQubit.range(n_qubits)
    circuit = cirq.Circuit()
    circuit.append(cirq.H(qubits[0]))
    for i in range(0, n_qubits - 1):
        circuit.append(cirq.CNOT(qubits[i], qubits[i + 1]))

    return_type = "cirq" if not return_type else return_type

    return convert_from_mitiq(circuit, return_type)
