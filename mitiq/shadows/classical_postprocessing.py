# Copyright (C) Unitary Fund
# Portions of this code have been adapted from PennyLane's tutorial
# on Classical Shadows.
# Original authors: PennyLane developers: Brian Doolittle, Roeland Wiersema
# Tutorial link: https://pennylane.ai/qml/demos/tutorial_classical_shadows
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.
"""Classical post-processing process of classical shadows."""

from collections import defaultdict
from functools import reduce
from itertools import compress
from operator import mul
from statistics import median
from typing import Any, Dict, List, Optional, Tuple

import cirq
import numpy as np
import numpy.typing as npt

import mitiq
from mitiq.shadows.shadows_utils import (
    batch_calibration_data,
    create_string,
    valid_bitstrings,
)
from mitiq.utils import matrix_kronecker_product, operator_ptm_vector_rep

# Local unitaries to measure Pauli operators in the Z basis
PAULI_MAP = {
    "X": cirq.unitary(cirq.H),
    "Y": cirq.unitary(cirq.H) @ cirq.unitary(cirq.S).conj(),
    "Z": cirq.unitary(cirq.I),
}

# Density matrices of single-qubit basis states
ZERO_STATE = np.diag([1.0 + 0.0j, 0.0 + 0.0j])
ONE_STATE = np.diag([0.0 + 0.0j, 1.0 + 0.0j])


def get_single_shot_pauli_fidelity(
    bitstring: str, paulistring: str, locality: Optional[int] = None
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
    pauli_fidelity = {"Z0": 1.0, "Z1": -1.0}
    local_fidelities = [
        pauli_fidelity.get(p + b, 0.0) for b, p in zip(bitstring, paulistring)
    ]
    num_qubits = len(bitstring)
    bitstrings = valid_bitstrings(num_qubits, max_hamming_weight=locality)
    fidelities = {}
    for bitstring in bitstrings:
        subset_fidelities = compress(local_fidelities, map(int, bitstring))
        fidelities[bitstring] = reduce(mul, subset_fidelities, 1.0)

    return fidelities


def get_pauli_fidelities(
    calibration_outcomes: Tuple[List[str], List[str]],
    batch_size: int,
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
    means = defaultdict(list)
    for bitstrings, paulistrings in batch_calibration_data(
        calibration_outcomes, batch_size
    ):
        all_fidelities = defaultdict(list)
        for bitstring, paulistring in zip(bitstrings, paulistrings):
            fidelities = get_single_shot_pauli_fidelity(
                bitstring, paulistring, locality=locality
            )
            for b, f in fidelities.items():
                all_fidelities[b].append(f)

        for bitstring, fids in all_fidelities.items():
            means[bitstring].append(sum(fids) / batch_size)

    return {
        bitstring: median(averages) for bitstring, averages in means.items()
    }


def classical_snapshot(
    b_list_shadow: str,
    u_list_shadow: str,
    f_est: Optional[Dict[str, float]] = None,
) -> npt.NDArray[Any]:
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
        f_est: The estimated Pauli fidelities to use for calibration if
            available.

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

    if f_est:
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
            local_rho = 3.0 * (U.conj().T @ state @ U) - cirq.unitary(cirq.I)
            local_rhos.append(local_rho)

        rho_snapshot = matrix_kronecker_product(local_rhos)
    return rho_snapshot


def shadow_state_reconstruction(
    shadow_measurement_outcomes: Tuple[List[str], List[str]],
    fidelities: Optional[Dict[str, float]] = None,
) -> npt.NDArray[Any]:
    """Reconstruct a state approximation as an average over all snapshots.

    Args:
        shadow_measurement_outcomes: Measurement result and the basis
            performing the measurement obtained from `random_pauli_measurement`
            for classical shadow protocol.
        f_est: The estimated Pauli fidelities to use for calibration if
            available.
    Returns:
        The state reconstructed from classical shadow protocol
    """
    bitstrings, paulistrings = shadow_measurement_outcomes

    return np.mean(
        [
            classical_snapshot(bitstring, paulistring, fidelities)
            for bitstring, paulistring in zip(bitstrings, paulistrings)
        ],
        axis=0,
    )


def expectation_estimation_shadow(
    measurement_outcomes: Tuple[List[str], List[str]],
    pauli_str: mitiq.PauliString,
    k_shadows: int,
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
        f_est: The estimated Pauli fidelities to use for calibration if
            available.

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

    # classical values stored in classical computer
    b_lists_shadow = np.array([list(u) for u in measurement_outcomes[0]])[
        :, target_locs
    ]
    u_lists_shadow = np.array([list(u) for u in measurement_outcomes[1]])[
        :, target_locs
    ]

    means = []

    # loop over the splits of the shadow:
    group_idxes = np.array_split(np.arange(len(b_lists_shadow)), k_shadows)

    # loop over the splits of the shadow:
    for idxes in group_idxes:
        matching_indexes = np.nonzero(
            np.all(u_lists_shadow[idxes] == target_obs, axis=1)
        )

        if len(matching_indexes[0]):
            product = (-1) ** np.sum(
                b_lists_shadow[idxes][matching_indexes].astype(int),
                axis=1,
            )

            if f_est:
                b = create_string(num_qubits, target_locs)
                f_val = f_est.get(b, np.inf)
                # product becomes an array of snapshot expectation values
                # witch satisfy condition (1) and (2)
                product = (1 / f_val) * product
            else:
                product = 3 ** len(target_locs) * product

        else:
            product = 0.0

        # append the mean of the product in each split
        means.append(np.sum(product) / len(idxes))

    # return the median of means
    return float(np.real(np.median(means) * coeff))
