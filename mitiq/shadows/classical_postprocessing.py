# Copyright (C) Unitary Fund
# Portions of this code have been adapted from PennyLane's tutorial
# on Classical Shadows.
# Original authors: PennyLane developers: Brian Doolittle, Roeland Wiersema
# Tutorial link: https://pennylane.ai/qml/demos/tutorial_classical_shadows
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.
"""Classical post-processing process of classical shadows."""

from itertools import combinations
from typing import Any, Dict, List, Optional, Tuple

import cirq
import numpy as np
from numpy.typing import NDArray

import mitiq
from mitiq.shadows.shadows_utils import bitstring_to_eigenvalues, create_string
from mitiq.utils import matrix_kronecker_product, operator_ptm_vector_rep

# Local unitaries to measure Pauli operators in the Z basis
PAULI_MAP = {
    "X": cirq.H._unitary_(),
    "Y": cirq.H._unitary_() @ cirq.S._unitary_().conj(),
    "Z": cirq.I._unitary_(),
}

# Density matrices of single-qubit basis states
ZERO_STATE = np.diag([1.0 + 0.0j, 0.0 + 0.0j])
ONE_STATE = np.diag([0.0 + 0.0j, 1.0 + 0.0j])

# F_LOCAL_MAP is based on local Pauli fidelity of qubit i
# f_b_i = <<b_i|U_i|P_z^b_i>>
# s.t. f_0 = U_11U_11^* + U_12U_12^*, f_1 = U_21U_21^* + U_22U_22^*
F_LOCAL_MAP = {
    "0X": 0.0,
    "0Y": 0.0,
    "0Z": 1.0,
    "1X": 0.0,
    "1Y": 0.0,
    "1Z": -1.0,
}

"""
The following functions are used in the classical post-processing
of calibration
"""


def get_single_shot_pauli_fidelity(
    bit_string: str,
    pauli_string: str,
    locality: Optional[int] = None,
) -> Dict[str, float]:
    r"""
    Calculate Pauli fidelity :math:`f_b` for a single shot measurement of the
    calibration circuit for b= bit_string.

    In the notation of arXiv:2011.09636, this function estimates the
    coefficient :math:`f_b`, which characterizes the (noisy) classical
    shadow channel.

    The locality is realized on the assumption that the noisy
    channel :math:`\Lambda` is local
    :math:`\Lambda \equiv \bigotimes_i^n\Lambda_i`.

    Args:
        bit_string: The bitstring corresponding to a computational basis state.
            E.g., '01...0':math:`:=|0\rangle|1\rangle...|0\rangle`.
        pauli_string: The local Pauli measurement performed on each qubit.
            e.g.'XY...Z' means perform local X-basis measurement on the
            1st qubit, local Y-basis measurement the 2ed qubit, local Z-basis
            measurement the last qubit in the circuit.
        locality: The locality of the operator, whose expectation value is
            going to be estimated by the classical shadow. E.g., if the
            operator is the Ising model Hamiltonian with nearest neighbor
            interactions, then locality = 2.

    Returns:
        A dictionary of Pauli fidelity bit_string: :math:`\{{f}_b\}`.
        If the locality is :math:`w < n`, then derive the output's keys from
        the bit_string. Ensure that the number of 1s in the keys is less
        than or equal to w. The corresponding Pauli fidelity is the product of
        local Pauli fidelity where the associated locus in the keys are '1'.
    """
    num_qubits = len(bit_string)
    if locality is None:
        locality = num_qubits
    # local_pauli_fidelity is a list of local Pauli fidelity for each qubit
    local_pauli_fidelity = np.array(
        [F_LOCAL_MAP[b + u] for b, u in zip(bit_string, pauli_string)]
    )
    # f_est is a dictionary of Pauli fidelity for each b_string
    f_est = {create_string(num_qubits, []): 1.0}
    for w in range(1, locality + 1):
        target_locs = np.array(list(combinations(range(num_qubits), w)))
        single_round_pauli_fidelity = np.prod(
            local_pauli_fidelity[target_locs], axis=1
        )
        for loc, fidelity in zip(target_locs, single_round_pauli_fidelity):
            # b_str is a string of length n with maximum number of w 1s.
            b_str = create_string(num_qubits, loc)
            f_est[b_str] = fidelity

    return f_est


def get_pauli_fidelities(
    calibration_measurement_outcomes: Tuple[List[str], List[str]],
    k_calibration: int,
    locality: Optional[int] = None,
) -> Dict[str, complex]:
    r"""
    Calculate Pauli fidelities for the calibration circuit. In the notation of
    arXiv:2011.09636, this function estimates the coefficients
    :math:`f_b`, which characterize the (noisy) classical shadow channel
    :math:`\mathcal{M}=\sum_b f_b \Pi_b`.

    Args:
        calibration_measurement_outcomes: The `random_Pauli_measurement`
            outcomes for the state :math:`|0\rangle^{\otimes n}`}` .
        k_calibration: number of splits in the median of means estimator.
        locality: The locality of the operator, whose expectation value is
            going to be estimated by the classical shadow. E.g., if the
            operator is the Ising model Hamiltonian with nearest neighbor
            interactions, then locality = 2.

    Returns:
        A :math:`2^n`-dimensional dictionary of Pauli fidelities
        :math:`f_b` for :math:`b = \{0,1\}^{n}`
    """

    # classical values of random Pauli measurement stored in classical computer
    b_lists, u_lists = calibration_measurement_outcomes

    # number of measurements in each split
    n_total_measurements = len(b_lists)

    means: Dict[str, List[float]] = {}  # key is bitstring, value is mean

    group_idxes = np.array_split(
        np.arange(n_total_measurements), k_calibration
    )
    # loop over the splits of the shadow:
    for idxes in group_idxes:
        b_lists_k = np.array(b_lists)[idxes]
        u_lists_k = np.array(u_lists)[idxes]

        n_group_measurements = len(b_lists_k)
        group_results = []
        for j in range(n_group_measurements):
            bitstring, u_list = b_lists_k[j], u_lists_k[j]
            f_est = get_single_shot_pauli_fidelity(
                bitstring, u_list, locality=locality
            )
            group_results.append(f_est)

        f_est_group = {
            b: sum([f[b] for f in group_results]) / n_group_measurements
            for b in group_results[0]
        }

        for bitstring, mean in f_est_group.items():
            if bitstring not in means:
                means[bitstring] = []
            means[bitstring].append(mean)
    # return the median of means
    medians = {
        bitstring: complex(np.median(values))
        for bitstring, values in means.items()
    }
    return medians


"""
The following functions are used in the classical post-processing of
classical shadows.
"""


def classical_snapshot(
    b_list_shadow: str,
    u_list_shadow: str,
    pauli_twirling_calibration: bool,
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
    # ptm rep of identity
    I_ptm = operator_ptm_vector_rep(np.eye(2) / np.sqrt(2))
    # define projections Pi_0 and Pi_1
    pi_zero = np.outer(I_ptm, I_ptm)
    pi_one = np.eye(4) - pi_zero
    pi_zero = np.diag(pi_zero)
    pi_one = np.diag(pi_one)

    if pauli_twirling_calibration:
        if f_est is None:
            raise ValueError(
                "estimation of Pauli fidelity must be provided for Pauli"
                "twirling calibration."
            )
        elements = []
        # get b_list and f for each calibration measurement
        for b_list_cal, f in f_est.items():
            pi_snapshot_vecter = []
            for b_1, b2, u2 in zip(b_list_cal, b_list_shadow, u_list_shadow):
                # get pi for each qubit based on calibration measurement
                pi = pi_zero if b_1 == "0" else pi_one
                # get state for each qubit based on shadow measurement
                state = ZERO_STATE if b2 == "0" else ONE_STATE
                # get U2 for each qubit based on shadow measurement
                U2 = PAULI_MAP[u2]
                pi_snapshot_vecter.append(
                    pi * operator_ptm_vector_rep(U2.conj().T @ state @ U2)
                )
                # solve for the snapshot state
            elements.append(
                1 / f * matrix_kronecker_product(pi_snapshot_vecter)
            )
        rho_snapshot_vector = np.sum(elements, axis=0)
        # normalize the snapshot state
        rho_snapshot = rho_snapshot_vector  # * normalize_factor
    # w/o calibration, noted here, the output in terms of matrix,
    # not in PTM rep.
    else:
        local_rhos = []
        for b, u in zip(b_list_shadow, u_list_shadow):
            state = ZERO_STATE if b == "0" else ONE_STATE
            U = PAULI_MAP[u]
            # apply inverse of the quantum channel,get PTM vector rep
            local_rho = 3.0 * (U.conj().T @ state @ U) - cirq.I._unitary_()
            local_rhos.append(local_rho)

        rho_snapshot = matrix_kronecker_product(local_rhos)
    return rho_snapshot


def shadow_state_reconstruction(
    shadow_measurement_outcomes: Tuple[List[str], List[str]],
    pauli_twirling_calibration: bool,
    f_est: Optional[Dict[str, float]] = None,
) -> NDArray[Any]:
    """Reconstruct a state approximation as an average over all snapshots.

    Args:
        shadow_measurement_outcomes: Measurement result and the basis
            performing the measurement obtained from `random_pauli_measurement`
            for classical shadow protocol.
        shadow_measurement_outcomes: Measurement results obtained from
            `random_pauli_measurement` for classical shadow protocol.
        pauli_twirling_calibration: Whether to use Pauli twirling
            calibration.
        f_est: The estimated Pauli fidelity for each calibration
    Returns:
        The state reconstructed from classical shadow protocol
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
    pauli_twirling_calibration: bool,
    f_est: Optional[Dict[str, float]] = None,
) -> float:
    """Calculate the expectation value of an observable from classical shadows.
    Use median of means to ameliorate the effects of outliers.

    Args:
        measurement_outcomes: A shadow tuple obtained from
            `z_basis_measurement`.
        pauli_str: Single mitiq observable consisting of
            Pauli operators.
        k_shadows: number of splits in the median of means estimator.
        pauli_twirling_calibration: Whether to use Pauli twirling
            calibration.
        f_est: The estimated Pauli fidelities for each calibration

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

                b = create_string(num_qubits, target_locs)
                f_val = f_est.get(b, None)
                if f_val is None:
                    means.append(0.0)
                else:
                    # product becomes an array of snapshots expectation values
                    # witch satisfy condition (1) and (2)
                    product = (1.0 / f_val) * product
            else:
                product = 3 ** (target_support.count("1")) * product

            # append the mean of the product in each split
            means.append(np.sum(product) / n_group_measurements)

    # return the median of means
    return float(np.real(np.median(means) * coeff))
