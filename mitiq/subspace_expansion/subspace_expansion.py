"""API for using Subspace Expansion error mitigation."""

from typing import Callable, Sequence, Union
from mitiq import Executor, Observable, QPROGRAM, QuantumResult, PauliString
import cirq
from scipy.linalg import eig

from mitiq.subspace_expansion.utils import (
    convert_from_cirq_PauliString_to_Mitiq_PauliString,
    convert_from_Mitiq_Observable_to_cirq_PauliSum,
)


cache = {}  # tuple of pauli mask: expectation value


def execute_with_subspace_expansion(
    circuit: QPROGRAM,
    executor: Union[Executor, Callable[[QPROGRAM], QuantumResult]],
    check_operators: Sequence[PauliString],
    code_hamiltonian: Observable,
    observable: Observable,
) -> QuantumResult:
    """Function for the calculation of an observable from some circuit of
        interest to be mitigated with Subspace Expansion
    Args:
        circuit: Quantum program to execute with error mitigation.
        executor: Executes a circuit and returns a `QuantumResult`.
        check_operators: List of check operators that define the stabilizer code space.
        code_hamiltonian: Hamiltonian of the code space.
        observable: Observable to compute the mitigated expectation value of.
    """
    cache.clear()
    check_operators_cirq = [
        check_operator._pauli for check_operator in check_operators
    ]
    code_hamiltonian_cirq = convert_from_Mitiq_Observable_to_cirq_PauliSum(
        code_hamiltonian
    )
    observable_cirq = convert_from_Mitiq_Observable_to_cirq_PauliSum(
        observable
    )
    P = get_projector(
        circuit, executor, check_operators_cirq, code_hamiltonian_cirq
    )
    pop = get_expectation_value_for_observable(
        circuit, executor, P * observable_cirq * P
    )
    pp = get_expectation_value_for_observable(circuit, executor, P * P)
    return pop / pp


def get_expectation_value_for_observable(
    qc: QPROGRAM,
    executor: Union[Executor, Callable[[QPROGRAM], QuantumResult]],
    observable: Union[cirq.PauliString, cirq.PauliSum],
):
    def get_expectation_value_for_one_pauli(pauli_string: cirq.PauliString):
        cache_key = tuple(pauli_string.gate.pauli_mask.tolist())
        if cache_key == ():  # if all I's
            return 1 * pauli_string.coefficient
        if cache_key in cache:
            return cache[cache_key] * pauli_string.coefficient
        mitiq_pauli_string = (
            convert_from_cirq_PauliString_to_Mitiq_PauliString(pauli_string)
        )
        return Observable(mitiq_pauli_string).expectation(qc, executor)

    total_expectation_value_for_observable = 0

    if type(observable) == cirq.PauliString:
        pauli_string = observable
        total_expectation_value_for_observable += (
            get_expectation_value_for_one_pauli(pauli_string)
        )
    elif type(observable) == cirq.PauliSum:
        pauli_sum = observable
        for pauli_string in pauli_sum:
            total_expectation_value_for_observable += (
                get_expectation_value_for_one_pauli(pauli_string)
            )
    return total_expectation_value_for_observable


def get_projector(qc: QPROGRAM, executor, check_operators, code_hamiltonian):
    S = compute_overlap_matrix(qc, executor, check_operators)
    H = compute_code_hamiltonian_overlap_matrix(qc, executor, check_operators, code_hamiltonian)
    E, C = eig(H, S)
    minpos = list(E).index(min(E))
    Cs = C[:, minpos]

    projector_formed_out_of_check_operators_and_Cs = sum(
        [check_operators[i] * Cs[i] for i in range(len(check_operators))]
    )

    return projector_formed_out_of_check_operators_and_Cs


def compute_overlap_matrix(
    qc: QPROGRAM,
    executor: Union[Executor, Callable[[QPROGRAM], QuantumResult]],
    check_operators: Sequence[cirq.PauliString],
):
    S = []
    # S_ij = <Ψ|Mi† Mj|Ψ>
    for i in range(len(check_operators)):
        S.append([])
        for j in range(len(check_operators)):
            paulis_and_weights = check_operators[i] * check_operators[j]
            sij = get_expectation_value_for_observable(
                qc, executor, paulis_and_weights
            )
            S[-1].append(sij)
    return S


def compute_code_hamiltonian_overlap_matrix(
    qc: QPROGRAM,
    executor: Union[Executor, Callable[[QPROGRAM], QuantumResult]],
    check_operators: Sequence[cirq.PauliString],
    code_hamiltonian,
):
    H = []
    # H_ij = <Ψ|Mi† H Mj|Ψ>
    for i in range(len(check_operators)):
        H.append([])
        for j in range(len(check_operators)):
            paulis_and_weights = (
                check_operators[i] * code_hamiltonian * check_operators[j]
            )
            H[-1].append(
                get_expectation_value_for_observable(
                    qc, executor, paulis_and_weights
                )
            )
    return H
