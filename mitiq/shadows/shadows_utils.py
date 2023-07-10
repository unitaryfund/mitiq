from typing import Tuple, List
import cirq
import numpy as np
from cirq.ops.pauli_string import PauliString


# given error of the fidelity between the true state and the reconstructed state,
# return the number of measurements needed
def min_n_total_measurements(epsilon: float, num_qubits: int) -> int:
    return int(34 * (4**num_qubits) * epsilon ** (-2))


# based on the theorem, we calculate N,K for the shadow bound
def calculate_shadow_bound(
    error: float, observables: List[PauliString], failure_rate: float
) -> Tuple[int, int]:
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


def operator_2_norm(R: np.ndarray[np.complex128]) -> float:
    return float(
        np.sqrt(np.trace(R.conjugate().transpose() @ R)).reshape(-1).real
    )


def fidelity(
    state_vector: np.ndarray[np.complex128], rho: np.ndarray[np.complex128]
) -> float:
    return float(
        np.reshape(state_vector.conj().T @ rho @ state_vector, -1).real
    )
