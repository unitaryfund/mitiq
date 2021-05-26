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

"""Utilities related to probabilistic error cancellation."""

from typing import List
from itertools import product
from copy import deepcopy
import numpy as np

from cirq import (
    Circuit,
    OP_TREE,
    H,
    CNOT,
    LineQubit,
    DensityMatrixSimulator,
    DepolarizingChannel,
    AmplitudeDampingChannel,
    channel,
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


def _circuit_to_choi(circuit: Circuit) -> np.ndarray:
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


def _operation_to_choi(operation_tree: OP_TREE) -> np.ndarray:
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


def _tensor_product_kraus(
    local_kraus: List[np.ndarray], num_channels: int
) -> List[np.ndarray]:
    """Given the a local channel (defined by its kraus operators),
    this functions returns the Kraus operators corresponding to tensor
    product of "num_channels" copies of the input channel.
    """
    tensored_kraus = []
    for kraus_tuple in product(local_kraus, repeat=num_channels):
        kraus_product = np.eye(1)
        for kraus in kraus_tuple:
            kraus_product = np.kron(kraus_product, kraus)
        tensored_kraus.append(kraus_product)
    return tensored_kraus


def global_depolarizing_kraus(
    noise_level: float, num_qubits: int
) -> List[np.ndarray]:
    """Returns the kraus operators of a global depolarizing channel
    at a given noise level.
    """
    noisy_op = DepolarizingChannel(noise_level, num_qubits)
    return list(channel(noisy_op))


def local_depolarizing_kraus(
    noise_level: float, num_qubits: int
) -> List[np.ndarray]:
    """Returns the Kraus operators of the tensor product of local
    depolarizing channels acting on each qubit.
    """
    local_kraus = global_depolarizing_kraus(noise_level, num_qubits=1)
    return _tensor_product_kraus(local_kraus, num_channels=num_qubits)


def amplitude_damping_kraus(
    noise_level: float, num_qubits: int
) -> List[np.ndarray]:
    """Returns the Kraus operators of an amplitude damping
    channel at a given noise level. If num_qubits > 1, the Kraus operators
    corresponding to tensor product of many single-qubit amplitude damping
    channels are returned.
    """
    noisy_op = AmplitudeDampingChannel(noise_level)
    local_kraus = list(channel(noisy_op))
    return _tensor_product_kraus(local_kraus, num_channels=num_qubits)


def matrix_to_vector(density_matrix: np.ndarray) -> np.array:
    """Reshapes a density matrix into a vector, according to the rule:
    |i><j| --> |i,j>.
    """
    return density_matrix.flatten()


def vector_to_matrix(vector: np.ndarray) -> np.array:
    """Reshapes a vector to a density matrix, according to the rule:
    |i,j> --> |i><j|.
    """
    dim = int(np.round(np.sqrt(vector.size)))
    return vector.reshape(dim, dim)


def kraus_to_super(kraus_ops: List[np.ndarray]) -> np.ndarray:
    """Maps a set of Kraus operators into a single superoperator
    matrix "S" acting by matrix multiplication on vectorized
    density matrices.

    The returned matrix "S" is obtained with the formula:

    S = \\sum_j K_j \\otimes K_j^*,

    where K_j are the superoperators.
    The mapping is based on the following isomorphism:

    A|i><j|B  <=>  (A \\otimes B^T) |i>|j>.
    """
    terms_to_sum = [np.kron(k, k.conj()) for k in kraus_ops]
    return sum(terms_to_sum)


def choi_to_super(choi_state: np.ndarray) -> np.ndarray:
    """Returns the superoperator matrix corresponding to
    the channel defined by the input (normalized) Choi state.

    Up to normalization, this is just a tensor transposition.
    """
    dim = int(np.round(np.sqrt(choi_state.shape[0])))
    choi_kl_ij = choi_state.reshape(dim, dim, dim, dim)
    choi_ki_lj = choi_kl_ij.transpose(0, 2, 1, 3)
    super_not_normalized = choi_ki_lj.reshape(dim ** 2, dim ** 2)
    return dim * super_not_normalized


def super_to_choi(super_operator: np.ndarray) -> np.ndarray:
    """Returns the normalized choi state corresponding to
    the channel defined by the input superoperator.

    Up to normalization, this is just a tensor transposition.
    """
    dim_squared = super_operator.shape[0]
    # We use that a transposition is a self-inverse operation
    return choi_to_super(super_operator) / dim_squared
