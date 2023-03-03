# Copyright (C) 2023 Unitary Fund
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

""" Functions for creating 2 W-state benchmarking circuits as defined in
:cite:`Cruz_2019_Efficient`"""

from typing import Optional
import numpy as np
import cirq

from mitiq import QPROGRAM
from mitiq.interface import convert_from_mitiq


# Build a linear complexity circuit
def W_circuit_linear_complexity(
    n_qubits: int,
    return_type: Optional[str] = None,
) -> QPROGRAM:
    """Returns a circuit to create a ``n_qubit`` qubit W-state with linear
    complexity.
    Args:
        n_qubits: The number of qubits in the circuit.
        return_type: Return type of the output circuit.
    Returns:
        A W-state circuit of linear complexity acting on ``n_qubits`` qubits.
    """
    if n_qubits <= 0:
        raise ValueError("{} is invalid for the number of qubits. ", n_qubits)

    qubits = cirq.LineQubit.range(n_qubits)
    circuit = cirq.Circuit()

    for i, j in zip(range(0, n_qubits), range(1, n_qubits)):
        N = n_qubits - i
        circuit.append(
            cirq.Ry(rads=1 / N * np.pi).controlled().on(qubits[i], qubits[j])
        )
        circuit.append(cirq.CNOT(qubits[j], qubits[i]))

    return_type = "cirq" if not return_type else return_type

    return convert_from_mitiq(circuit, return_type)


# Logarithmic time complexity circuit
