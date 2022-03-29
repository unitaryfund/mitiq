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


def _validate_integer_matrix(mask: np.ndarray) -> None:
    """Ensures the input is a NumPy 2d array with integer elements."""
    if not isinstance(mask, np.ndarray):
        raise TypeError("The input matrix must be a numpy.ndarray object.")
    if not np.issubdtype(mask.dtype.type, int):
        raise TypeError("The input matrix must have integer elements.")
    if len(mask.shape) != 2:
        raise ValueError("The input must be a 2-dimensional array.")


def get_slack_matrix_from_circuit_mask(mask: np.ndarray) -> np.ndarray:
    """Given a circuit mask matrix A, e.g. the output of get_circuit_mask(),
    returns a slack matrix B, where B_{i,j} = t if the position A{i,j} is the
    initial element of a sequence of t zeros (from left to right).

    Args:
        mask: The mask matrix of a quantum circuit.

    Returns: The matrix of slack lengths.
    """
    _validate_integer_matrix(mask)
    if not (mask**2 == mask).all():
        raise ValueError("The input matrix elements must be 0 or 1.")

    num_rows, num_cols = mask.shape
    slack_matrix = np.zeros((num_rows, num_cols), dtype=int)
    for r in range(num_rows):
        for c in range(num_cols):
            previous_elem = mask[r, c - 1] if c != 0 else 1
            if previous_elem == 1:
                # Compute slack length
                for elem in mask[r, c::]:
                    if elem == 0:
                        slack_matrix[r, c] += 1
                    else:
                        break

    return slack_matrix
