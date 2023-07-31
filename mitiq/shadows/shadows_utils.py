# Copyright (C) Unitary Fund
# Portions of this code have been adapted from PennyLane's tutorial
# on Classical Shadows.
# Original authors: PennyLane developers: Brian Doolittle, Roeland Wiersema
# Tutorial link: https://pennylane.ai/qml/demos/tutorial_classical_shadows
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

"""Defines utility functions for classical shadows protocol."""
from typing import Tuple, List, Any

import numpy as np
from numpy.typing import NDArray
import cirq

import mitiq

from itertools import product

Paulis = [
    cirq.I._unitary_(),
    cirq.X._unitary_(),
    cirq.Y._unitary_(),
    cirq.Z._unitary_(),
]


def kronecker_product(matrices: List[NDArray[Any]]) -> NDArray[Any]:
    """
    Returns the Kronecker product of a list of matrices.
    """
    result = matrices[0]
    for matrix in matrices[1:]:
        result = np.kron(result, matrix)
    return result


# Input must be a 2x2 matrix, output is a vector of 4 elements
def operator_ptm_vector_rep(O: NDArray[Any]) -> NDArray[Any]:
    r"""
    Returns the PTM vector representation :math:`|o\rangle\!\rangle\in \mathcal{H}_{4^n}`
    of an operator :math:`o\in \mathcal{L}(\mathcal{H}_{2^n})`.
    """
    # vector i-th entry is math:`d^{-1/2}Tr(oP_i)`
    # where P_i is the i-th Pauli matrix
    assert (
        len(O.shape) == 2 and O.shape[0] == O.shape[1]
    ), "Input must be a square matrix"
    num_qubits = int(np.log2(O.shape[0]))
    O_vec = []
    for pauli_combination in product(Paulis, repeat=num_qubits):
        kron_product = kronecker_product(pauli_combination)
        O_vec.append(np.trace(O @ kron_product) * np.sqrt(1 / 2**num_qubits))
    return np.array(O_vec)


def eigenvalues_to_bitstring(values: List[int]) -> str:
    """Converts eigenvalues to bitstring. e.g., [-1,1,1] -> '100'"""
    return "".join(["1" if v == -1 else "0" for v in values])


def bitstring_to_eigenvalues(bitstring: str) -> List[int]:
    """Converts bitstring to eigenvalues. e.g., '100' -> [-1,1,1]"""
    return [1 if b == "0" else -1 for b in bitstring]


def create_string(n: int, loc_list: List[int]) -> str:
    """
    This function returns a string of length n with 1s at the locations
    specified by loc_list and 0s elsewhere.
    """
    loc_set = set(loc_list)  # Convert list to set for efficient lookups
    return "".join(map(lambda i: "1" if i in loc_set else "0", range(n)))


def n_measurements_tomography_bound(epsilon: float, num_qubits: int) -> int:
    """
    This function returns the minimum number of classical shadows required
    for state reconstruction for achieving the desired accuracy.

    Args:
        epsilon: The error on the estimator.
        num_qubits: The number of qubits in the system.

    Returns:
        An integer that gives the number of snapshots required to satisfy the
        shadow bound.
    """
    return int(34 * (4**num_qubits) * epsilon ** (-2))


def local_clifford_shadow_norm(obs: mitiq.PauliString) -> float:
    """
    Calculate shadow norm of an operator with random unitary sampled from local
    Clifford group.
    Args:
        opt: a self-adjoint operator.
    Returns:
        Shadow norm when unitary ensemble is local Clifford group.
    """
    opt = obs.matrix()
    norm = (
        np.linalg.norm(
            opt - np.trace(opt) / 2 ** int(np.log2(opt.shape[0])),
            ord=np.inf,
        )
        ** 2
    )
    return float(norm)


def n_measurements_opts_expectation_bound(
    error: float,
    observables: List[mitiq.PauliString],
    failure_rate: float,
) -> Tuple[int, int]:
    """
    This function returns the minimum number of classical shadows required and
    the number of groups "k" into which we need to split the shadows for
    achieving the desired accuracy and failure rate in operator expectation
    value estimation.

    Args:
        error: The error on the estimator.
        observables: List of mitiq.PauliString corresponding to the
            observables we intend to measure.
        failure_rate: Rate of failure for the bound to hold.

    Returns:
        Integers quantifying the number of snapshots required to satisfy
        the shadow bound and the chunk size required to attain the specified
        failure rate.
    """
    M = len(observables)
    K = 2 * np.log(2 * M / failure_rate)

    N = (
        34
        * max(local_clifford_shadow_norm(o) for o in observables)
        / error**2
    )
    return int(np.ceil(N * K)), int(K)


def fidelity(
    state_vector: NDArray[np.complex64],
    rho: NDArray[np.complex64],
) -> float:
    """
    Calculate the fidelity between a state vector and a density matrix.
    Args:
        state_vector: The vector whose norm we want to calculate.
        rho: The operator whose norm we want to calculate.

    Returns:
        Scalar corresponding to the fidelity.
    """
    return np.reshape(state_vector.conj().T @ rho @ state_vector, -1).real[0]
