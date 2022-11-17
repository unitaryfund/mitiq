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
from cirq import Circuit, LineQubit, synchronize_terminal_measurements
import numpy as np
import numpy.typing as npt
from typing import Callable
from mitiq.interface import noise_scaling_converter
from mitiq import QPROGRAM


def _get_circuit_mask(circuit: Circuit) -> npt.NDArray[np.int64]:
    """Given a circuit with n qubits and d moments returns a matrix
    :math:`A` with n rows and d columns. The matrix elements are
    :math:`A_{i,j} = 1` if there is a gate acting on qubit :math:`i` at moment
    :math:`j`, while :math:`A_{i,j} = 0` otherwise.

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


def _validate_integer_matrix(mask: npt.NDArray[np.int64]) -> None:
    """Ensures the input is a NumPy 2d array with integer elements."""
    if not isinstance(mask, np.ndarray):
        raise TypeError("The input matrix must be a numpy.ndarray object.")
    if not np.issubdtype(mask.dtype.type, int):
        raise TypeError("The input matrix must have integer elements.")
    if len(mask.shape) != 2:
        raise ValueError("The input must be a 2-dimensional array.")


def get_slack_matrix_from_circuit_mask(
    mask: npt.NDArray[np.int64],
) -> npt.NDArray[np.int64]:
    """Given a circuit mask matrix :math:`A`, e.g., the output of
    ``_get_circuit_mask()``, returns a slack matrix :math:`B`,
    where :math:`B_{i,j} = t` if the position :math:`A_{i,j}` is the
    initial element of a sequence of :math:`t` zeros (from left to right).

    Args:
        mask: The mask matrix of a quantum circuit.

    Returns:
        The matrix of slack lengths.
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


def insert_ddd_sequences(
    circuit: QPROGRAM,
    rule: Callable[[int], Circuit],
) -> QPROGRAM:
    """Returns the circuit with DDD sequences applied according to the input
    rule.

    Args:
        circuit: The QPROGRAM circuit to be modified with DDD sequences.
        rule: The rule determining what DDD sequences should be applied.
            A set of built-in DDD rules can be imported from
            ``mitiq.ddd.rules``.

    Returns:
        The circuit with DDD sequences added.
    """

    return _insert_ddd_sequences(circuit, rule)


@noise_scaling_converter
def _insert_ddd_sequences(
    circuit: Circuit,
    rule: Callable[[int], Circuit],
) -> Circuit:
    """Returns the circuit with DDD sequences applied according to the input
    rule.

    Args:
        circuit: The Cirq circuit to be modified with DDD sequences.
        rule: The rule determining what DDD sequences should be applied.
            A set of built-in DDD rules can be imported from
            ``mitiq.ddd.rules``.

    Returns:
        The circuit with DDD sequences added.
    """
    circuit = synchronize_terminal_measurements(circuit)
    if not circuit.are_all_measurements_terminal():
        raise ValueError(
            "This circuit contains midcircuit measurements which "
            "are not currently supported by DDD."
        )

    slack_matrix = get_slack_matrix_from_circuit_mask(
        _get_circuit_mask(circuit)
    )
    # Copy to avoid mutating the input circuit
    circuit_with_ddd = circuit.copy()
    qubits = sorted(circuit.all_qubits())
    for moment_idx in range(len(circuit)):
        slack_column = slack_matrix[:, moment_idx]
        for row_index, slack_length in enumerate(slack_column):
            if slack_length > 1:
                ddd_sequence = rule(slack_length).transform_qubits(
                    {LineQubit(0): qubits[row_index]}
                )
                for idx, op in enumerate(ddd_sequence.all_operations()):
                    circuit_with_ddd._moments[
                        moment_idx + idx
                    ] = circuit_with_ddd._moments[
                        moment_idx + idx
                    ].with_operations(
                        op
                    )
    return circuit_with_ddd
