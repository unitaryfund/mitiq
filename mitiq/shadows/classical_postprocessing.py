# Copyright (C) Unitary Fund
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.
"""Classical post-processing process of classical shadows."""

from typing import Tuple, List, Any, Union

import cirq
import numpy as np
from numpy.typing import NDArray
from mitiq.shadows.utils import test_transform_to_cirq_paulistring

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
        from `z_basis_measurement`.

    Returns:
        Numpy array with the reconstructed quantum state.
    """
    num_snapshots, num_qubits = measurement_outcomes[0].shape

    # classical values
    b_lists, u_lists = measurement_outcomes

    # Averaging over snapshot states.
    shadow_rho = np.sum(
        [
            classical_snapshot(b_list, u_list)
            for b_list, u_list in zip(b_lists, u_lists)
        ],
        axis=0,
    )

    return shadow_rho / num_snapshots


def expectation_estimation_shadow(
    measurement_outcomes: Tuple[NDArray[Any], NDArray[np.string_]],
    pauli_string: Union[str, cirq.PauliString, mitiq.PauliString],  # type: ignore
    k_shadows: int,
) -> float:
    """Calculate the expectation value of an observable from classical shadows.
    Use median of means to ameliorate the effects of outliers.

    Args:
        measurement_outcomes: A shadow tuple obtained from
            `random_pauli_measurement`.
        pauli_string: Single cirq observable consisting of
            Pauli operators.
        k_shadows: number of splits in the median of means estimator.

    Returns:
        Estimation of the observable expectation value.
    """
    pauli_string = transform_to_cirq_paulistring(pauli_string)

    # target observable
    target_obs, target_locs = [], []
    for qubit, pauli in pauli_string.items():
        target_obs.append(str(pauli))
        target_locs.append(int(qubit))

    b_lists, u_lists = measurement_outcomes
    u_lists = np.array([list(u) for u in u_lists])

    n_total_measurements = len(b_lists)
    means = []

    # loop over the splits of the shadow:
    for i in range(0, n_total_measurements, n_total_measurements // k_shadows):
        # assign the splits temporarily
        b_lists_k, u_lists_k = (
            b_lists[i : i + n_total_measurements // k_shadows],
            u_lists[i : i + n_total_measurements // k_shadows],
        )
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
    return float(np.median(means))
