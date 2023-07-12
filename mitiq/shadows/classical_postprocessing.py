from typing import Tuple, List, Union, Any

import numpy as np

PAULI_MAP = {"X": 0, "Y": 1, "Z": 2}


def snapshot_state(
    b_list: List[bool], u_list: List[Union[int, float]]
) -> np.ndarray:
    """Impliment a single snapshot state reconstruction,

    Args:
        b_list (array): The list of classical outcomes for the snapshot.
        u_list (array): Indices for the applied Pauli measurement.

    Returns:
        Numpy array with the reconstructed snapshot.
    """
    num_qubits = len(b_list)

    # computational basis states, e.g. $$|0\rangle=(1,0)$$, and zero_state$$=|0\rangle\langle0|$$.
    zero_state = np.array([[1, 0], [0, 0]])
    one_state = np.array([[0, 0], [0, 1]])

    # local unitaries that applied to the computational basis states, e.g. $$Z-$$basis measurement,
    # which equivalent to a random Pauli measurement, i.e. for each qubit, we randomly decide to measure the Pauli operators.
    phase_z = np.array([[1, 0], [0, -1j]], dtype=complex)
    hadamard = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    identity = np.eye(2)

    # act $$Ad_{U^\dagger}$$ on the computational basis states, where $$U$$ is the local unitary that used to do the Pauli measurement.
    unitaries = [hadamard, hadamard @ phase_z, identity]

    # reconstructing a single snapshot state by applying Eq. (S44)
    rho_snapshot = [1]
    for i in range(num_qubits):
        state = zero_state if b_list[i] == 1 else one_state
        U = unitaries[PAULI_MAP[u_list[i]]]

        # applying Eq. (S44)
        local_rho = 3 * (U.conj().T @ state @ U) - identity
        rho_snapshot = np.kron(rho_snapshot, local_rho)

    return rho_snapshot


def shadow_state_reconstruction(
    measurement_outcomes: Tuple[np.ndarray, np.ndarray]
) -> np.ndarray:
    """Reconstruct a state approximation as an average over all snapshots.

    Args:
        shadow (tuple): A shadow tuple obtained from `z_basis_measurement`.

    Returns:
        Numpy array with the reconstructed quantum state.
    """
    num_snapshots, num_qubits = measurement_outcomes[0].shape

    # classical values
    b_lists, u_lists = measurement_outcomes

    # Averaging over snapshot states.
    shadow_rho = np.zeros((2**num_qubits, 2**num_qubits), dtype=complex)
    for i in range(num_snapshots):
        shadow_rho += snapshot_state(b_lists[i], u_lists[i])

    return shadow_rho / num_snapshots


def expectation_estimation_shadow(
    measurement_outcomes: Tuple[np.ndarray, np.ndarray],
    observable: Any,
    k: int,
) -> float:
    # Need R = NK in total, and split into K subsets of size and N is the number of snapshots.
    # each subset is a tuple of (b_lists, u_lists) and each element of the list is a list of length len(qubits)

    r"""Calculate the estimator $$E[O] = median(Tr{rho_{(k)} O})$$ where
    $$rho_(k))$$is set of $$k$$ snapshots in the shadow. Use median of means to
    ameliorate the effects of outliers.

    Args:
        measurement_outcomes (tuple): A shadow tuple obtained from `get_z_basis_measurement`.
        observable (cirq.PauliString): Single cirq observable consisting of single Pauli
            operators e.g. cirq.X(0) * cirq.Y(1).
        k (int): number of splits in the median of means estimator. k * N = R, where R is the total number of measurements, N is the number of snapshots.

    Returns:
        Scalar corresponding to the estimate of the observable.
    """

    # convert cirq observables to indices
    # map_pauli_to_int = {cirq.X: "X", cirq.Y: 1, cirq.Z: 2}
    target_obs, target_locs = [], []
    for qubit, pauli in observable.items():
        target_obs.append(str(pauli))
        target_locs.append(int(qubit))

    # classical values stored in classical computer
    b_lists, u_lists = measurement_outcomes
    u_lists = np.array([list(u) for u in u_lists])
    # number of measurements in each split
    n_total_measurements = len(b_lists)
    means = []

    # loop over the splits of the shadow:
    for i in range(0, n_total_measurements, n_total_measurements // k):
        # assign the splits temporarily
        b_lists_k, u_lists_k = (
            b_lists[i : i + n_total_measurements // k],
            u_lists[i : i + n_total_measurements // k],
        )
        n_group_measurements = len(b_lists_k)
        # find the exact matches for the observable of interest at the specified locations
        indices = np.all(u_lists_k[:, target_locs] == target_obs, axis=1)

        # catch the edge case where there is no match in the chunk
        if sum(indices) > 0:
            # take the product and sum
            product = np.prod(3 * (b_lists_k[indices][:, target_locs]), axis=1)
            means.append(np.sum(product) / n_group_measurements)
        else:
            means.append(0)
    return np.median(means)
