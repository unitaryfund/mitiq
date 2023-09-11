# Copyright (C) Unitary Fund
# Portions of this code have been adapted from PennyLane's tutorial
# on Classical Shadows.
# Original authors: PennyLane developers: Brian Doolittle, Roeland Wiersema
# Tutorial link: https://pennylane.ai/qml/demos/tutorial_classical_shadows
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

"""Defines utility functions for classical shadows protocol."""

from typing import Iterable, List, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import sqrtm

import mitiq


def eigenvalues_to_bitstring(values: Iterable[int]) -> str:
    """Converts eigenvalues to bitstring. e.g., ``[-1,1,1] -> "100"``

    Args:
        values: A list of eigenvalues (must be $-1$ and $1$).

    Returns:
        A string of 1s and 0s corresponding to the states associated to
        eigenvalues.
    """
    return "".join(["1" if v == -1 else "0" for v in values])


def bitstring_to_eigenvalues(bitstring: str) -> List[int]:
    """Converts bitstring to eigenvalues. e.g., ``"100" -> [-1,1,1]``

    Args:
        bitstring: A string of 1s and 0s.

    Returns:
        A list of eigenvalues (either $-1$ or $1$) corresponding to the
        bitstring.
    """
    return [1 if b == "0" else -1 for b in bitstring]


def create_string(str_len: int, loc_list: List[int]) -> str:
    """
    This function returns a string of length ``str_len`` with 1s at the
    locations specified by ``loc_list`` and 0s elsewhere.

    Args:
        str_len: The length of the string.
        loc_list: A list of integers indices specifying the locations of 1s in
            the string.
    Returns:
        A bitstring constructed as above.

    Example:
        A basic example::

            create_string(5, [1, 3])
            >>> "01010"
    """
    return "".join(
        map(lambda i: "1" if i in set(loc_list) else "0", range(str_len))
    )


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
        obs: A self-adjoint operator, i.e. mitiq.PauliString with real coffe.
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
    sigma: NDArray[np.complex64], rho: NDArray[np.complex64]
) -> float:
    """
    Calculate the fidelity between two states.

    Args:
        sigma: A state in terms of square matrix or vector.
        rho: A state in terms square matrix or vector.

    Returns:
        Scalar corresponding to the fidelity.
    """
    if sigma.ndim == 1 and rho.ndim == 1:
        val = np.abs(np.dot(sigma.conj(), rho)) ** 2.0
    elif sigma.ndim == 1 and rho.ndim == 2:
        val = np.abs(sigma.conj().T @ rho @ sigma)
    elif sigma.ndim == 2 and rho.ndim == 1:
        val = np.abs(rho.conj().T @ sigma @ rho)
    elif sigma.ndim == 2 and rho.ndim == 2:
        val = np.abs(np.trace(sqrtm(sigma) @ rho @ sqrtm(sigma)))
    else:
        raise ValueError("Invalid input dimensions")
    return float(val)
