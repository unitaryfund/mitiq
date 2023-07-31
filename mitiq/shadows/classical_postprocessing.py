# Copyright (C) Unitary Fund
# Portions of this code have been adapted from PennyLane's tutorial
# on Classical Shadows.
# Original authors: PennyLane developers: Brian Doolittle, Roeland Wiersema
# Tutorial link: https://pennylane.ai/qml/demos/tutorial_classical_shadows
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.
"""Classical post-processing process of classical shadows."""

from typing import Tuple, List, Any, Dict, Optional
import cirq
import numpy as np
from numpy.typing import NDArray
import mitiq
from mitiq.shadows.shadows_utils import (
    bitstring_to_eigenvalues,
    create_string,
    operator_ptm_vector_rep,
    kronecker_product,
)


# local unitary that applied to the qubits
phase_z = cirq.S._unitary_().conj()
hadamard = cirq.H._unitary_()
identity = cirq.I._unitary_()
PAULI_MAP = {"X": hadamard, "Y": hadamard @ phase_z, "Z": identity}

# z-basis measurement outcomes to 4d-vector representation
b_zero = np.array([[1.0 + 0.0j, 0.0 + 0.0j]])
zero_state = b_zero.T @ b_zero

b_one = np.array([[0.0 + 0.0j, 1.0 + 0.0j]])
one_state = b_one.T @ b_one

"""
The following functions are used in the classical post-processing
of calibration
"""


def get_single_shot_pauli_fidelity(
    b_list: List[str], u_list: List[str]
) -> float:
    r"""
    Implement a single snapshot state reconstruction.

    Args:
        b_list: Circuit qubits computational basis
            e.g. :math:`"01..":=|0\rangle|1\rangle..`.

        u_list: The local Pauli measurement performed on each qubit. e.g."XY.."
            means perform local X-basis measurement on the 1st qubit,
             local Y-basis measurement the 2ed qubit in the circuit.

    Returns:
        Calculate Pauli fidelity for a single shot, and return the result as a
        numpy array: :math:`\{\hat{f}^{(r)}_{m_i}\}_{i\leq n}` with n is the
        number of qubits in the circuit.
    """
    single_round_pauli_fidelity = 1.0

    for b, u in zip(b_list, u_list):
        # b_state = zero_state if m == 1 else one_state
        # U = PAULI_MAP[u]
        # p_m = cirq.I._unitary_() if m == 1 else cirq.Z._unitary_()

        # local_pauli_fidelity = np.abs(np.trace(b_state.conj().T @ U @ p_m))

        b_state = 0 if b == "0" else 1
        U = PAULI_MAP[u]
        P_z = cirq.Z._unitary_()
        local_pauli_fidelity = (
            2.0 * U[b_state, :] * P_z * (U.conj().T[:, b_state])
        )
        local_pauli_fidelity = local_pauli_fidelity[0, 0]

        single_round_pauli_fidelity *= local_pauli_fidelity

    return single_round_pauli_fidelity


def get_pauli_fidelity(
    calibration_measurement_outcomes: Tuple[List[str], List[str]],
    k_calibration: int,
) -> Dict[str, complex]:
    r"""
    Calculate Pauli fidelity for a perticular

    Args:
        calibration_measurement_outcomes: The `random_Pauli_measurement`
            outcomes with circuit :math:`|0\rangle^{\otimes n}`}`
        k_calibration: number of splits in the median of means estimator.

    Returns:
        an :math:`2^n`-dimensional array of Pauli fidelity
        :math:'\hat{f}_m' for :math:`m = \{0,1\}^{n}`
    """

    # classical values of random Pauli measurement stored in classical computer
    b_lists, u_lists = calibration_measurement_outcomes

    # number of measurements in each split
    n_total_measurements = len(b_lists)

    means: Dict[str, List[float]] = {
        bitstring: [] for bitstring in b_lists
    }  # key is bitstring, value is mean

    group_idxes = np.array_split(
        np.arange(n_total_measurements), k_calibration
    )
    # loop over the splits of the shadow:
    for idxes in group_idxes:
        b_lists_k = np.array(b_lists)[idxes]
        u_lists_k = np.array(u_lists)[idxes]

        n_group_measurements = len(b_lists_k)
        group_results = {}
        for j in range(n_group_measurements):
            bitstring, u_list = b_lists_k[j], u_lists_k[j]
            f_est = get_single_shot_pauli_fidelity(bitstring, u_list)
            # if not exist,create a new key;
            # otherwise,append to the existing key
            if bitstring not in group_results:
                group_results[bitstring] = [f_est]
            else:
                group_results[bitstring].append(f_est)

        for bitstring in means.keys():
            if bitstring not in group_results:
                mean = 0.0
            else:
                mean = sum(group_results[bitstring]) / n_group_measurements
            means[bitstring].append(mean)
    # return the median of means
    medians = {
        bitstring: complex(np.median(values))
        for bitstring, values in means.items()
    }
    return medians


# calculate trace(pi_b-1^{}*pi_b)
def get_normalize_factor(
    f_est: Dict[str, float],
) -> float:
    num_qubits = len(list(f_est.keys())[0])
    trace_pi_b = 0.0
    for b_list_cal, f in f_est.items():
        trace_pi_b += 1 / f * 3 ** b_list_cal.count("1")

        # get normalize factor of inverse quantum channel
    return 10**num_qubits / trace_pi_b


"""
The following functions are used in the classical post-processing of
classical shadows.
"""
# ptm rep of identity
I_ptm = operator_ptm_vector_rep(np.eye(2) / np.sqrt(2))
# define projections Pi_0 and Pi_1
pi_zero = np.outer(I_ptm, I_ptm)
pi_one = np.eye(4) - pi_zero
pi_zero = np.diag(pi_zero)
pi_one = np.diag(pi_one)


def classical_snapshot(
    b_list_shadow: str,
    u_list_shadow: str,
    pauli_twirling_calibration: bool = False,
    f_est: Optional[Dict[str, float]] = None,
) -> NDArray[Any]:
    r"""
    Implement a single snapshot state reconstruction
    with calibration of the noisy quantum channel.

    Args:
        b_list_shadow: The list of length 1, classical outcomes for the
            snapshot. Here,
            b = '0' corresponds to :math:`|0\rangle`, and
            b = '1' corresponds to :math:`|1\rangle`.
        u_list_shadow: list of len 1, contains str of ("XYZ..") for
            the applied Pauli measurement on each qubit.
        pauli_twirling_calibration: Whether to use Pauli twirling
            calibration.
        f_est: The estimated Pauli fidelity for each calibration

    Returns:
        Reconstructed classical snapshot in terms of nparray.
    """
    # calibrate the noisy quantum channel, output in PTM rep.
    if pauli_twirling_calibration:
        if f_est is None:
            raise ValueError(
                "estimation of Pauli fidelity must be provided for Pauli"
                "twirling calibration."
            )
        elements = []
        normalize_factor = get_normalize_factor(f_est)
        # get b_list and f for each calibration measurement
        for b_list_cal, f in f_est.items():
            pi_snapshot_vecter = []
            for b_1, b2, u2 in zip(b_list_cal, b_list_shadow, u_list_shadow):
                # get pi for each qubit based on calibration measurement
                pi = pi_zero if b_1 == "0" else pi_one
                # get state for each qubit based on shadow measurement
                state = zero_state if b2 == "0" else one_state
                # get U2 for each qubit based on shadow measurement
                U2 = PAULI_MAP[u2]
                pi_snapshot_vecter.append(
                    pi * operator_ptm_vector_rep(U2.conj().T @ state @ U2)
                )
                # solve for the snapshot state
            elements.append(1 / f * kronecker_product(pi_snapshot_vecter))
        rho_snapshot_vector = np.sum(elements, axis=0)
        # normalize the snapshot state
        rho_snapshot = rho_snapshot_vector * normalize_factor
    # w/o calibration, noted here, the output in terms of matrix,
    # not in PTM rep.
    else:
        local_rhos = []
        for b, u in zip(b_list_shadow, u_list_shadow):
            state = zero_state if b == "0" else one_state
            U = PAULI_MAP[u]
            # apply inverse of the quantum channel,get PTM vector rep
            local_rho = 3.0 * (U.conj().T @ state @ U) - identity
            local_rhos.append(local_rho)

        rho_snapshot = kronecker_product(local_rhos)
    return rho_snapshot


def shadow_state_reconstruction(
    shadow_measurement_outcomes: Tuple[List[str], List[str]],
    pauli_twirling_calibration: bool = False,
    f_est: Optional[Dict[str, float]] = None,
) -> NDArray[Any]:
    """Reconstruct a state approximation as an average over all snapshots.

    Args:
        shadow_measurement_outcomes: Measurement result and the basis
            performing the measurement obtained from
            `random_pauli_measurement` for classical shadow protocol.
        pauli_twirling_calibration: Whether to use Pauli twirling
            calibration.
        f_est: The estimated Pauli fidelity for each calibration
    Returns:
        Numpy array with the reconstructed state.
    """
    # classical values of random Pauli measurement stored in classical computer
    b_lists_shadow, u_lists_shadow = shadow_measurement_outcomes

    # Averaging over snapshot states.
    return np.mean(
        [
            classical_snapshot(
                b_list_shadow, u_list_shadow, pauli_twirling_calibration, f_est
            )
            for b_list_shadow, u_list_shadow in zip(
                b_lists_shadow, u_lists_shadow
            )
        ],
        axis=0,
    )


def expectation_estimation_shadow(
    measurement_outcomes: Tuple[List[str], List[str]],
    pauli_str: mitiq.PauliString,  # type: ignore
    k_shadows: int,
    pauli_twirling_calibration: bool = False,
    f_est: Optional[Dict[str, float]] = None,
) -> float:
    """Calculate the expectation value of an observable from classical shadows.
    Use median of means to ameliorate the effects of outliers.

    Args:
        measurement_outcomes: A shadow tuple obtained from
            `z_basis_measurement`.
        observable: Single cirq observable consisting of
            Pauli operators.
        k_shadows: number of splits in the median of means estimator.

    Returns:
        Float corresponding to the estimate of the observable
        expectation value.
    """
    num_qubits = len(measurement_outcomes[0][0])
    obs = pauli_str._pauli
    coeff = pauli_str.coeff

    target_obs, target_locs = [], []
    for qubit, pauli in obs.items():
        target_obs.append(str(pauli))
        target_locs.append(int(qubit))
    # if target_locs is [2,3,4], then target_support is "001110"
    target_support = create_string(num_qubits, target_locs)

    # classical values stored in classical computer
    b_lists_shadow = np.array(measurement_outcomes[0])
    u_lists_shadow = np.array([list(u) for u in measurement_outcomes[1]])
    # number of measurements in each split
    n_total_measurements_shadow = len(b_lists_shadow)
    means = []
    # loop over the splits of the shadow:
    group_idxes = np.array_split(
        np.arange(n_total_measurements_shadow), k_shadows
    )
    # loop over the splits of the shadow:
    for idxes in group_idxes:
        b_lists_shadow_k = b_lists_shadow[idxes]
        u_lists_shadow_k = u_lists_shadow[idxes]
        # number of measurements/shadows in each split
        n_group_measurements = len(b_lists_shadow_k)

        # observable is    obs = {IIXZYI}, or target_loc =[2,3,4],
        # then the non-zero elements in product should satisfies (1)
        # and (2) if calibration is used

        # (1)shadow measure U_2 = {..XZY.}, exactly match
        indices = np.all(
            u_lists_shadow_k[:, target_locs] == target_obs, axis=1
        )
        if sum(indices) == 0:
            means.append(0.0)
        else:
            eigenvalues = np.array(
                [
                    bitstring_to_eigenvalues(b)
                    for b in b_lists_shadow_k[indices]
                ]
            )
            product = np.prod(eigenvalues[:, target_locs], axis=1)

            if pauli_twirling_calibration:
                if f_est is None:
                    raise ValueError(
                        "estimation of Pauli fidelity must be provided for"
                        "Pauli twirling calibration."
                    )
                normalize_factor = get_normalize_factor(f_est)
                # (2)the cali b_list_cal={"001110"} should exactly match the
                # target_support = "001110"
                # indices_Pi_obs is a boolean array of length 2^num_qubits
                indices_Pi_obs = np.array(list(f_est.keys())) == target_support
                if sum(indices_Pi_obs) == 0:
                    means.append(0.0)
                else:
                    # product becomes an array of snapshots expectation values
                    # witch satisfy condition (1) and (2)
                    product = (
                        np.sum(
                            (1 / np.array(list(f_est.values())))
                            * indices_Pi_obs
                        )
                        * product
                    )
            else:
                product = 3 ** (target_support.count("1")) * product

            # append the mean of the product in each split
            means.append(np.sum(product) / n_group_measurements)

    # return the median of means
    return float(np.real(np.median(means) * coeff))
