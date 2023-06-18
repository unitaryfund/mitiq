# Copyright (C) 2021 Unitary Fund
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""High-level Quantum Susbapce Expansion tools."""

from typing import Callable, Sequence, Dict, List
from mitiq import Executor, Observable, QPROGRAM, QuantumResult, PauliString
from functools import wraps
from .qse_utils import get_projector, get_expectation_value_for_observable


def execute_with_qse(
    circuit: QPROGRAM,
    executor: Callable[[QPROGRAM], QuantumResult],
    check_operators: Sequence[PauliString],
    code_hamiltonian: Observable,
    observable: Observable,
    pauli_string_to_expectation_cache: Dict[PauliString, complex] = {},
) -> float:
    """Function for the calculation of an observable from some circuit of
        interest to be mitigated with Quantum Subspace Expansion
    Args:
        circuit: Quantum program to execute with error mitigation.
        executor: Executes a circuit and returns a `QuantumResult`.
        check_operators: List of check operators that define the
        stabilizer code space.
        code_hamiltonian: Hamiltonian of the code space.
        observable: Observable to compute the mitigated expectation value of.
        pauli_string_to_expectation_cache: Cache for expectation values of
        Pauli strings used to compute the projector and the observable.
    Returns:
        The expectation value estimated with QSE.
    """
    P = get_projector(
        circuit,
        executor,
        check_operators,
        code_hamiltonian,
        pauli_string_to_expectation_cache,
    )
    # Compute the expectation value of the observable: <P O P>
    pop = get_expectation_value_for_observable(
        circuit,
        executor,
        P * observable * P,
        pauli_string_to_expectation_cache,
    )
    # Compute the normalization factor: <P P>
    pp = get_expectation_value_for_observable(
        circuit, executor, P * P, pauli_string_to_expectation_cache
    )
    return pop / pp


def mitigate_executor(
    executor: Callable[[QPROGRAM], QuantumResult],
    check_operators: Sequence[PauliString],
    code_hamiltonian: Observable,
    observable: Observable,
    pauli_string_to_expectation_cache: Dict[PauliString, complex] = {},
) -> Callable[[QPROGRAM], float]:
    """Returns a modified version of the input 'executor' which is
    error-mitigated with zero-noise extrapolation (ZNE).

    Args:
        circuit: Quantum program to execute with error mitigation.
        executor: Executes a circuit and returns a `QuantumResult`.
        check_operators: List of check operators that define the
        stabilizer code space.
        code_hamiltonian: Hamiltonian of the code space.
        observable: Observable to compute the mitigated expectation value of.
        pauli_string_to_expectation_cache: Cache for expectation values of
        Pauli strings used to compute the projector and the observable.
        share_cache: Only applicable for batched executors. If True, the
        cache is shared between the all circuits in the batch.
    Returns:
        The error-mitigated version of the input executor.
    """
    executor_obj = Executor(executor)
    if not executor_obj.can_batch:

        @wraps(executor)
        def new_executor(circuit: QPROGRAM) -> float:
            return execute_with_qse(
                circuit,
                executor,
                check_operators,
                code_hamiltonian,
                observable,
                pauli_string_to_expectation_cache,
            )

    else:

        @wraps(executor)
        def new_executor(circuits: List[QPROGRAM]) -> List[float]:
            return [
                execute_with_qse(
                    circuit,
                    executor,
                    check_operators,
                    code_hamiltonian,
                    observable,
                    pauli_string_to_expectation_cache,
                )
                for circuit in circuits
            ]

    return new_executor


def qse_decorator(
    check_operators: Sequence[PauliString],
    code_hamiltonian: Observable,
    observable: Observable,
    pauli_string_to_expectation_cache: Dict[PauliString, complex] = {},
) -> Callable[
    [Callable[[QPROGRAM], QuantumResult]], Callable[[QPROGRAM], float]
]:
    """Decorator which adds an error-mitigation layer based on quantum
    subspace expansion (QSE) to an executor function, i.e., a function which
    executes a quantum circuit with an arbitrary backend and returns a
    ``QuantumResult``.

    Args:
        check_operators: List of check operators that define the
        stabilizer code space.
        code_hamiltonian: Hamiltonian of the code space.
        observable: Observable to compute the mitigated expectation value of.
        pauli_string_to_expectation_cache: Cache for expectation values of
        Pauli strings used to compute the projector and the observable.
        share_cache: Only applicable for batched executors. If True, the
        cache is shared between the all circuits in the batch.

    Returns:
        The error-mitigating decorator to be applied to an executor function.
    """

    def decorator(
        executor: Callable[[QPROGRAM], QuantumResult]
    ) -> Callable[[QPROGRAM], float]:
        val = mitigate_executor(
            executor,
            check_operators,
            code_hamiltonian,
            observable,
            pauli_string_to_expectation_cache,
        )
        return val

    return decorator
