# Copyright (C) 2020 Unitary Fund
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

# TODO: Functions which don't fit in pec.py and sampling.py are placed here.
#  Some of them could be moved in future new sub-modules of PEC
#  (e.g. decomposition, tomo, etc.)

"""Utilities for manipulating matrix representations of quantum channels."""

from typing import List
from copy import deepcopy
import numpy as np
import numpy.typing as npt

from cirq import (
    Circuit,
    OP_TREE,
    H,
    CNOT,
    LineQubit,
    DensityMatrixSimulator,
)


def _max_ent_state_circuit(num_qubits: int) -> Circuit:
    r"""Generates a circuit which prepares the maximally entangled state
    |\omega\rangle = U |0\rangle  = \sum_i |i\rangle \otimes |i\rangle .

    Args:
        num_qubits: The number of qubits on which the circuit is applied.
            It must be an even number because of the structure of a
            maximally entangled state.

    Returns:
        The circuits which prepares the state |\omega\rangle.

    Raises:
        Value error: if num_qubits is not an even positive integer.
    """

    if not isinstance(num_qubits, int) or num_qubits % 2 or num_qubits == 0:
        raise ValueError(
            "The argument 'num_qubits' must be an even and positive integer."
        )

    alice_reg = LineQubit.range(num_qubits // 2)
    bob_reg = LineQubit.range(num_qubits // 2, num_qubits)

    return Circuit(
        # Prepare alice_register in a uniform superposition
        H.on_each(*alice_reg),
        # Correlate alice_register with bob_register
        [CNOT.on(alice_reg[i], bob_reg[i]) for i in range(num_qubits // 2)],
    )


def _circuit_to_choi(circuit: Circuit) -> npt.NDArray[np.complex64]:
    """Returns the density matrix of the Choi state associated to the
    input circuit.

    The density matrix completely characterizes the quantum channel induced by
    the input circuit (including the effect of noise if present).

    Args:
        circuit: The input circuit.
    Returns:
        The density matrix of the Choi state associated to the input circuit.
    """
    simulator = DensityMatrixSimulator()
    num_qubits = len(circuit.all_qubits())
    # Copy and remove all operations
    full_circ = deepcopy(circuit)[0:0]
    full_circ += _max_ent_state_circuit(2 * num_qubits)
    full_circ += circuit
    return simulator.simulate(full_circ).final_density_matrix  # type: ignore


def _operation_to_choi(operation_tree: OP_TREE) -> npt.NDArray[np.complex64]:
    """Returns the density matrix of the Choi state associated to the
    input operation tree (e.g. a single operation or a sequence of operations).

    The density matrix completely characterizes the quantum channel induced by
    the input operation tree (including the effect of noise if present).

    Args:
        operation_tree: Nested list of operations.
    Returns:
        The density matrix of the Choi state associated to the input circuit.
    """
    circuit = Circuit(operation_tree)
    return _circuit_to_choi(circuit)


def tensor_product(
    *args: npt.NDArray[np.complex64],
) -> npt.NDArray[np.complex64]:
    """Returns the Kronecker product of the input array-like arguments.
    This is a generalization of the binary function
    ``numpy.kron(arg_a, arg_b)`` to the case of an arbitrary number of
    arguments.
    """
    if args == ():
        raise TypeError("tensor_product() requires at least one argument.")

    val = args[0]
    for term in args[1:]:
        val = np.kron(val, term)
    return val


def matrix_to_vector(
    density_matrix: npt.NDArray[np.complex64],
) -> npt.NDArray[np.complex64]:
    r"""Reshapes a :math:`d \times d` density matrix into a
    :math:`d^2`-dimensional state vector, according to the rule:
    :math:`|i \rangle\langle j| \rightarrow |i,j \rangle`.
    """
    return density_matrix.flatten()


def _safe_sqrt(
    perfect_square: int,
    error_str: str = "The input must be a square number.",
) -> int:
    """Takes the square root of the input integer and
    raises an error if the input is not a perfect square."""
    square_root = int(np.round(np.sqrt(perfect_square)))
    if square_root**2 != perfect_square:
        raise ValueError(error_str)
    return square_root


def vector_to_matrix(
    vector: npt.NDArray[np.complex64],
) -> npt.NDArray[np.complex64]:
    r"""Reshapes a :math:`d^2`-dimensional state vector into a
    :math:`d \times d` density matrix, according to the rule:
    :math:`|i,j \rangle \rightarrow |i \rangle\langle j|`.
    """
    error_str = (
        "The expected dimension of the input vector must be a"
        f" square number but is {vector.size}."
    )
    dim = _safe_sqrt(vector.size, error_str)
    return vector.reshape(dim, dim)


def kraus_to_super(
    kraus_ops: List[npt.NDArray[np.complex64]],
) -> npt.NDArray[np.complex64]:
    r"""Maps a set of Kraus operators into a single superoperator
    matrix acting by matrix multiplication on vectorized
    density matrices.

    The returned matrix :math:`S` is obtained with the formula:

    .. math::
        S = \sum_j K_j \otimes K_j^*,

    where :math:`\{K_j\}` are the Kraus operators.
    The mapping is based on the following isomorphism:

    .. math::
        A|i \rangle\langle  j|B  <=>  (A \otimes B^T) |i\rangle|j\rangle.
    """
    return np.array(sum(np.kron(k, k.conj()) for k in kraus_ops))


def choi_to_super(
    choi_state: npt.NDArray[np.complex64],
) -> npt.NDArray[np.complex64]:
    """Returns the superoperator matrix corresponding to
    the channel defined by the input (normalized) Choi state.

    Up to normalization, this is just a tensor transposition.
    """
    error_str = (
        "The expected dimension of the input matrix must be a"
        f" square number but is {choi_state.shape[0]}."
    )
    dim = _safe_sqrt(choi_state.shape[0], error_str)

    choi_kl_ij = choi_state.reshape(dim, dim, dim, dim)
    choi_ki_lj = choi_kl_ij.transpose(0, 2, 1, 3)
    super_not_normalized = choi_ki_lj.reshape(dim**2, dim**2)
    return dim * super_not_normalized


def super_to_choi(
    super_operator: npt.NDArray[np.complex64],
) -> npt.NDArray[np.complex64]:
    """Returns the normalized choi state corresponding to
    the channel defined by the input superoperator.

    Up to normalization, this is just a tensor transposition.
    """
    dim_squared = super_operator.shape[0]
    # We use that a transposition is a self-inverse operation
    return choi_to_super(super_operator) / dim_squared


def kraus_to_choi(
    kraus_ops: List[npt.NDArray[np.complex64]],
) -> npt.NDArray[np.complex64]:
    """Returns the normalized choi state corresponding to
    the channel defined by the input kraus operators.
    """
    return super_to_choi(kraus_to_super(kraus_ops))
