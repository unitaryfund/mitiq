# Copyright (C) Unitary Fund
# Portions of this code have been adapted from PennyLane's tutorial
# on Classical Shadows.
# Original authors: PennyLane developers: Brian Doolittle, Roeland Wiersema
# Tutorial link: https://pennylane.ai/qml/demos/tutorial_classical_shadows
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.
"""Classical post-processing process of classical shadows."""

from typing import Tuple, List, Any

import cirq
import numpy as np
from numpy.typing import NDArray

import mitiq

# local unitary that applied to the qubits
phase_z = cirq.S._unitary_().conj()
hadamard = cirq.H._unitary_()
identity = cirq.I._unitary_()
PAULI_MAP = {"X": hadamard, "Y": hadamard @ phase_z, "Z": identity}


def classical_snapshot(b_list: List[int], u_list: List[str]) -> NDArray[Any]:
    r"""
    Implement a single snapshot state reconstruction.

    Args:
        b_list: The list of classical outcomes for the snapshot. Here,
            b = 1 corresponds to :math:`|0\rangle`, and
            b = -1 corresponds to :math:`|1\rangle`.

        u_list: Array of ("X", "Y", "Z") for the applied
            Pauli measurement on each qubit.

    Returns:
        Reconstructed snapshot in terms of nparray, which is not a physical
        state.
    """

    # z-basis measurement outcomes
    b_zero = np.array([[1.0 + 0.0j, 0.0 + 0.0j]])
    zero_state = b_zero.T @ b_zero

    b_one = np.array([[0.0 + 0.0j, 1.0 + 0.0j]])
    one_state = b_one.T @ b_one

    rho_snapshot = np.array([1.0 + 0.0j])

    for b, u in zip(b_list, u_list):
        state = zero_state if b == 1 else one_state
        U = PAULI_MAP[u]
        # act $$Ad_{U^\dagger}$$ on the computational basis states
        local_rho = 3.0 * (U.conj().T @ state @ U) - identity
        rho_snapshot = np.kron(rho_snapshot, local_rho)

    return rho_snapshot


def shadow_state_reconstruction(
    measurement_outcomes: Tuple[NDArray[Any], NDArray[np.string_]]
) -> NDArray[Any]:
    """Reconstruct a state approximation as an average over all snapshots.

    Args:
        measurement_outcomes: A shadow tuple obtained
        from `random_pauli_measurement`.

    Returns:
        Numpy array with the reconstructed quantum state.
    """

    # classical values
    b_lists, u_lists = measurement_outcomes

    # Averaging over snapshot states.
    shadow_rho = np.mean(
        [
            classical_snapshot(b_list, u_list)
            for b_list, u_list in zip(b_lists, u_lists)
        ],
        axis=0,
    )

    return shadow_rho


def expectation_estimation_shadow(
    measurement_outcomes: Tuple[NDArray[Any], NDArray[np.string_]],
    pauli_str: mitiq.PauliString,
    k_shadows: int,
) -> complex:
    """Calculate the expectation value of an observable from classical shadows.
    Use median of means to ameliorate the effects of outliers.

    Args:
        measurement_outcomes: A shadow tuple obtained from
            `random_pauli_measurement`.
        pauli_str: Single observable consisting of Pauli operators.
        k_shadows: number of splits in the median of means estimator.

    Returns:
        Estimation of the observable expectation value.
    """
    # mitiq version
    obs = pauli_str._pauli
    coeff = pauli_str.coeff

    target_obs, target_locs = [], []
    for qubit, pauli in obs.items():
        target_obs.append(str(pauli))
        target_locs.append(int(qubit))

    b_lists, u_lists = measurement_outcomes
    u_lists = np.array([list(u) for u in u_lists])

    n_total_measurements = len(b_lists)
    means = []

    group_idxes = np.array_split(np.arange(n_total_measurements), k_shadows)
    # loop over the splits of the shadow:
    for idxes in group_idxes:
        b_lists_k = b_lists[idxes]
        u_lists_k = u_lists[idxes]
        # number of measurements/shadows in each split
        n_group_measurements = len(b_lists_k)
        # find the exact matches for the observable of
        # interest at the specified locations
        indices = np.all(u_lists_k[:, target_locs] == target_obs, axis=1)

        # catch the edge case where there is no match in the chunk
        if sum(indices) > 0:
            # take the product and sum
            product = np.prod(
                3.0 * (b_lists_k[indices][:, target_locs]), axis=1
            )
            means.append(np.sum(product) / n_group_measurements)
        else:
            means.append(0.0)
    # return the median of means
    return float(np.median(means)) * coeff
