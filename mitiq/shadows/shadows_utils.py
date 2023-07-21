# Copyright (C) Unitary Fund
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

"""Defines utility functions for classical shadows protocol."""
from typing import Tuple, List, Any

import cirq
import numpy as np
from cirq.ops.pauli_string import PauliString
from numpy.typing import NDArray


def min_n_total_measurements(epsilon: float, num_qubits: int) -> int:
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


def calculate_shadow_bound(
    error: float,
    observables: List[PauliString[Any]],
    failure_rate: float,
) -> Tuple[int, int]:
    """
    This function returns the minimum number of classical shadows required and
    the number of groups "k" into which we need to split the shadows for
    achieving the desired accuracy and failure rate in operator expectation
    value estimation.

    Args:
        error: The error on the estimator.
        observables: List of cirq.PauliString corresponding to the
        observables we intend to measure.
        failure_rate: Rate of failure for the bound to hold.

    Returns:
        Integers quantifying the number of snapshots required to satisfy
        the shadow bound and the chunk size required to attain the specified
        failure rate.
    """
    M = len(observables)
    K = 2 * np.log(2 * M / failure_rate)

    shadow_norm = (
        lambda opt: np.linalg.norm(
            cirq.unitary(opt)
            - np.trace(cirq.unitary(opt))
            / 2 ** int(np.log2(cirq.unitary(opt).shape[0])),
            ord=np.inf,
        )
        ** 2
    )
    N = 34 * max(shadow_norm(o) for o in observables) / error**2
    return int(np.ceil(N * K)), int(K)


def fidelity(
    state_vector: NDArray[Any],
    rho: NDArray[Any],
) -> float:
    """
    Calculate the fidelity between a state vector and a density matrix.
    Args:
        state_vector: The vector whose norm we want to calculate.
        rho: The operator whose norm we want to calculate.

    Returns:
        Scalar corresponding to the fidelity.
    """
    return float(
        np.reshape(state_vector.conj().T @ rho @ state_vector, -1).real
    )
