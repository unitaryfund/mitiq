"""API for using Clifford Data Regression (CDR) error mitigation."""

from typing import Any, Callable, Optional, Sequence, Union, List
from functools import wraps
from mitiq import Executor, Observable, QPROGRAM, QuantumResult, PauliString
import cirq
import sympy as sp
import numpy as np
from scipy.linalg import eig
from scipy.sparse.linalg import eigs

from typing import Union
from mitiq.subspace_expansion.utils import (
    convert_from_cirq_PauliString_to_Mitiq_PauliString,
    convert_from_Mitiq_Observable_to_cirq_PauliSum,
)


# TODO: move this to a singeleton

cache = {}  # tuple of pauli mask: expectation value


def execute_with_subspace_expansion(
    circuit: QPROGRAM,
    executor: Union[Executor, Callable[[QPROGRAM], QuantumResult]],
    Ms: list[cirq.PauliString],
    Hc: Observable,
    observable: Observable,
    **kwargs: Any,
) -> QuantumResult:
    """Function for the calculation of an observable from some circuit of
        interest to be mitigated with Subspace Expansion
    Args:
        circuit: Quantum program to execute with error mitigation.
        executor: Executes a circuit and returns a `QuantumResult`.
        observable: Observable to compute the expectation value of.
            If None, the `executor` must return an expectation value. Otherwise
            the `QuantumResult` returned by `executor` is used to compute the
            expectation of the observable.
        simulator: Executes a circuit without noise and returns a
            `QuantumResult`. For CDR to be efficient, the simulator must
            be able to efficiently simulate near-Clifford circuits.
    """
    cache.clear()
    Ms_cirq = [M._pauli for M in Ms]
    Hc_cirq = convert_from_Mitiq_Observable_to_cirq_PauliSum(Hc)
    observable_cirq = convert_from_Mitiq_Observable_to_cirq_PauliSum(
        observable
    )
    P = get_projector(circuit, executor, Ms_cirq, Hc_cirq)
    pop = get_expectation_value_for_observable(
        circuit, executor, P * observable_cirq * P
    )
    pp = get_expectation_value_for_observable(circuit, executor, P * P)
    return pop / pp


def get_expectation_value_for_observable(
    qc: QPROGRAM,
    executor: Union[Executor, Callable[[QPROGRAM], QuantumResult]],
    observable,
):
    def get_expectation_value_for_one_pauli(pauli_string: cirq.PauliString):
        cache_key = tuple(pauli_string.gate.pauli_mask.tolist())
        if cache_key == ():  # if all I's
            return 1 * pauli_string.coefficient
        if cache_key in cache:
            return cache[cache_key] * pauli_string.coefficient
        a = convert_from_cirq_PauliString_to_Mitiq_PauliString(pauli_string)
        return Observable(a).expectation(qc, executor)

    if type(observable) != list:
        observable = [observable]
    total_expectation_value_for_observable = 0
    for pauli_string_or_pauli_sum in observable:
        if type(pauli_string_or_pauli_sum) == cirq.PauliString:
            pauli_string = pauli_string_or_pauli_sum
            total_expectation_value_for_observable += (
                get_expectation_value_for_one_pauli(pauli_string)
            )
        elif type(pauli_string_or_pauli_sum) == cirq.PauliSum:
            pauli_sum = pauli_string_or_pauli_sum
            for pauli_string in pauli_sum:
                total_expectation_value_for_observable += (
                    get_expectation_value_for_one_pauli(pauli_string)
                )
    return total_expectation_value_for_observable


def get_Sij(qc: QPROGRAM, executor, Ms: list[cirq.PauliString]):
    Sij = []
    # Sij = <Ψ|Mi† Mj|Ψ>
    for i in range(len(Ms)):
        Sij.append([])
        for j in range(len(Ms)):
            paulis_and_weights = [Ms[i] * Ms[j]]
            sij = get_expectation_value_for_observable(
                qc, executor, paulis_and_weights
            )
            Sij[-1].append(sij)
    return Sij


def get_Hij(
    qc: QPROGRAM, executor, Ms: list[cirq.PauliString], Hc
):  # takes 40 minutes
    Hij = []
    # Hij = <Ψ|Mi† H Mj|Ψ>
    for i in range(len(Ms)):
        Hij.append([])
        for j in range(len(Ms)):
            paulis_and_weights = [Ms[i] * Hc * Ms[j]]
            Hij[-1].append(
                get_expectation_value_for_observable(
                    qc, executor, paulis_and_weights
                )
            )
    return Hij


def get_projector(qc: QPROGRAM, executor, Ms, Hc):
    Sij = get_Sij(qc, executor, Ms)
    Hij = get_Hij(qc, executor, Ms, Hc)
    E, C = eig(Hij, Sij)
    print("E=", E)
    minpos = list(E).index(min(E))
    print(minpos)
    Cs = C[:, minpos]

    projector_formed_out_of_Ms_and_Cs = sum(
        [Ms[i] * Cs[i] for i in range(len(Ms))]
    )

    return projector_formed_out_of_Ms_and_Cs
