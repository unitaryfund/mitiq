# Copyright (C) Unitary Fund
# Portions of this code have been adapted from PennyLane's tutorial
# on Classical Shadows.
# Original authors: PennyLane developers: Brian Doolittle, Roeland Wiersema
# Tutorial link: https://pennylane.ai/qml/demos/tutorial_classical_shadows
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

"""Defines utility functions for classical shadows protocol."""
from typing import Tuple, List

import numpy as np
from numpy.typing import NDArray

import mitiq


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
