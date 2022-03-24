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

"""Tools to determine slack windows in circuits and to insert DDD sequences."""
from cirq import Circuit
import numpy as np


def _get_circuit_mask(circuit: Circuit) -> np.ndarray:
    """Given a circuit with n qubits and d moments returns a matrix
    A with n rows and d columns. The matrix elements are A_{i,j} = 1 if
    there is a gate acting on qubit i at moment j, while A_{i,j} = 0 otherwise.

    Args:
        circuit: Input circuit to mask with n qubits and d moments

    Returns:
        A mask matrix with n rows and d columns
    """
    qubits = sorted(circuit.all_qubits())
    indexed_qubits = [(i, n) for (i, n) in enumerate(qubits)]
    mask_matrix = np.zeros((len(qubits), len(circuit)), dtype=int)
    for moment_index, moment in enumerate(circuit):
        for op in moment:
            qubit_indices = [
                qubit[0] for qubit in indexed_qubits if qubit[1] in op.qubits
            ]
            for qubit_index in qubit_indices:
                mask_matrix[qubit_index, moment_index] = 1
    return mask_matrix
