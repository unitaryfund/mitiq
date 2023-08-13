# Copyright (C) Unitary Fund
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

"""Functions for computing the projector for subspace expansion."""

from typing import Callable, Dict, List, Sequence, Union

import numpy as np
import numpy.typing as npt
from numpy.linalg import pinv
from scipy.linalg import eigh

from mitiq import QPROGRAM, Executor, Observable, PauliString, QuantumResult


def get_projector(
    circuit: QPROGRAM,
    executor: Union[Executor, Callable[[QPROGRAM], QuantumResult]],
    check_operators: Sequence[PauliString],
    code_hamiltonian: Observable,
    pauli_string_to_expectation_cache: Dict[PauliString, complex] = {},
) -> Observable:
    """Computes the projector onto the code space defined by the
    check_operators provided that minimizes the code_hamiltonian.

    Returns: Projector as an Observable.
    """
    S = _compute_overlap_matrix(
        circuit, executor, check_operators, pauli_string_to_expectation_cache
    )
    H = _compute_hamiltonian_overlap_matrix(
        circuit,
        executor,
        check_operators,
        code_hamiltonian,
        pauli_string_to_expectation_cache,
    )
    # We only want the smallest eigenvalue and corresponding eigenvector
    _, C = eigh(pinv(S) @ H, subset_by_index=[0, 0])
    # np float type: np.float64

    Cs = C[:, 0]
    projector = Observable(
        *[check_operators[i] * Cs[i] for i in range(len(check_operators))]
    )

    return projector


def get_expectation_value_for_observable(
    circuit: QPROGRAM,
    executor: Union[Executor, Callable[[QPROGRAM], QuantumResult]],
    observable: Union[PauliString, Observable],
    pauli_string_to_expectation_cache: Dict[PauliString, complex] = {},
) -> float:
    """Provide pauli_string_to_expectation_cache if you want to take advantage
    of caching.

    This function modifies pauli_string_to_expectation_cache in place.
    """

    def get_expectation_value_for_one_pauli(
        pauli_string: PauliString,
    ) -> float:
        cache_key = pauli_string.with_coeff(1)
        pauli_string_to_expectation_cache[cache_key] = final_executor.evaluate(
            circuit, Observable(cache_key)
        )[0]
        return (
            pauli_string_to_expectation_cache[cache_key] * pauli_string.coeff
        ).real

    total_expectation_value_for_observable = 0.0
    final_executor: Executor = (
        executor if isinstance(executor, Executor) else Executor(executor)
    )

    if isinstance(observable, PauliString):
        pauli_string = observable
        total_expectation_value_for_observable += (
            get_expectation_value_for_one_pauli(pauli_string)
        )
    elif isinstance(observable, Observable):
        for pauli_string in observable.paulis:
            total_expectation_value_for_observable += (
                get_expectation_value_for_one_pauli(pauli_string)
            )
    return total_expectation_value_for_observable


def _compute_overlap_matrix(
    circuit: QPROGRAM,
    executor: Union[Executor, Callable[[QPROGRAM], QuantumResult]],
    check_operators: Sequence[PauliString],
    pauli_string_to_expectation_cache: Dict[PauliString, complex] = {},
) -> npt.NDArray[np.float64]:
    S: List[List[float]] = []
    # S_ij = <Ψ|Mi† Mj|Ψ>
    for i in range(len(check_operators)):
        S.append([])
        for j in range(len(check_operators)):
            sij = get_expectation_value_for_observable(
                circuit,
                executor,
                check_operators[i] * check_operators[j],
                pauli_string_to_expectation_cache,
            )
            S[-1].append(sij)
    return np.array(S)


def _compute_hamiltonian_overlap_matrix(
    circuit: QPROGRAM,
    executor: Union[Executor, Callable[[QPROGRAM], QuantumResult]],
    check_operators: Sequence[PauliString],
    code_hamiltonian: Observable,
    pauli_string_to_expectation_cache: Dict[PauliString, complex] = {},
) -> npt.NDArray[np.float64]:
    H: List[List[float]] = []
    # H_ij = <Ψ|Mi† H Mj|Ψ>
    for i in range(len(check_operators)):
        H.append([])
        for j in range(len(check_operators)):
            H[-1].append(
                get_expectation_value_for_observable(
                    circuit,
                    executor,
                    check_operators[i] * code_hamiltonian * check_operators[j],
                    pauli_string_to_expectation_cache,
                )
            )
    return np.array(H)
