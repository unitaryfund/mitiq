# Copyright (C) 2020 Unitary Fund
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

"""High-level zero-noise extrapolation tools."""
from typing import Callable, Optional, Union, List
from functools import wraps

from mitiq import Executor, Observable, QPROGRAM, QuantumResult
from mitiq.zne.inference import Factory, RichardsonFactory
from mitiq.zne.scaling import fold_gates_at_random


def execute_with_zne(
    circuit: QPROGRAM,
    executor: Union[Executor, Callable[[QPROGRAM], QuantumResult]],
    observable: Optional[Observable] = None,
    *,
    factory: Optional[Factory] = None,
    scale_noise: Callable[[QPROGRAM, float], QPROGRAM] = fold_gates_at_random,
    num_to_average: int = 1,
) -> float:
    """Estimates the error-mitigated expectation value associated to the
    input circuit, via the application of zero-noise extrapolation (ZNE).

    Args:
        circuit: The input circuit to execute with ZNE.
        executor: A Mitiq executor that executes a circuit and returns the
            unmitigated ``QuantumResult`` (e.g. an expectation value).
        observable: Observable to compute the expectation value of. If
            ``None``, the ``executor`` must return an expectation value.
            Otherwise, the ``QuantumResult`` returned by ``executor`` is used
            to compute the expectation of the observable.
        factory: ``Factory`` object that determines the zero-noise
            extrapolation method.
        scale_noise: The function for scaling the noise of a quantum circuit.
            A list of built-in functions can be found in ``mitiq.zne.scaling``.
        num_to_average: Number of times expectation values are computed by
            the executor after each call to ``scale_noise``, then averaged.

    Returns:
        The expectation value estimated with ZNE.
    """
    if not factory:
        factory = RichardsonFactory(scale_factors=[1.0, 2.0, 3.0])

    if not isinstance(factory, Factory):
        raise TypeError(
            f"Argument `factory` must be of type mitiq.factories.Factory "
            f"but type(factory) is {type(factory)}."
        )

    if not callable(scale_noise):
        raise TypeError("Argument `scale_noise` must be callable.")

    if num_to_average < 1:
        raise ValueError("Argument `num_to_average` must be a positive int.")

    return factory.run(
        circuit, executor, observable, scale_noise, int(num_to_average)
    ).reduce()


def mitigate_executor(
    executor: Callable[[QPROGRAM], QuantumResult],
    observable: Optional[Observable] = None,
    *,
    factory: Optional[Factory] = None,
    scale_noise: Callable[[QPROGRAM, float], QPROGRAM] = fold_gates_at_random,
    num_to_average: int = 1,
) -> Callable[[QPROGRAM], float]:
    """Returns a modified version of the input 'executor' which is
    error-mitigated with zero-noise extrapolation (ZNE).

    Args:
        executor: A function that executes a circuit and returns the
            unmitigated `QuantumResult` (e.g. an expectation value).
        observable: Observable to compute the expectation value of. If None,
            the `executor` must return an expectation value. Otherwise,
            the `QuantumResult` returned by `executor` is used to compute the
            expectation of the observable.
        factory: Factory object determining the zero-noise extrapolation
            method.
        scale_noise: Function for scaling the noise of a quantum circuit.
        num_to_average: Number of times expectation values are computed by
            the executor after each call to scale_noise, then averaged.

    Returns:
        The error-mitigated version of the input executor.
    """
    executor_obj = Executor(executor)
    if not executor_obj.can_batch:

        @wraps(executor)
        def new_executor(circuit: QPROGRAM) -> float:
            return execute_with_zne(
                circuit,
                executor,
                observable,
                factory=factory,
                scale_noise=scale_noise,
                num_to_average=num_to_average,
            )

    else:

        @wraps(executor)
        def new_executor(circuits: List[QPROGRAM]) -> List[float]:
            return [
                execute_with_zne(
                    circuit,
                    executor,
                    observable,
                    factory=factory,
                    scale_noise=scale_noise,
                    num_to_average=num_to_average,
                )
                for circuit in circuits
            ]

    return new_executor


def zne_decorator(
    observable: Optional[Observable] = None,
    *,
    factory: Optional[Factory] = None,
    scale_noise: Callable[[QPROGRAM, float], QPROGRAM] = fold_gates_at_random,
    num_to_average: int = 1,
) -> Callable[
    [Callable[[QPROGRAM], QuantumResult]], Callable[[QPROGRAM], float]
]:
    """Decorator which adds an error-mitigation layer based on zero-noise
    extrapolation (ZNE) to an executor function, i.e., a function which
    executes a quantum circuit with an arbitrary backend and returns a
    ``QuantumResult`` (e.g. an expectation value).

    Args:
        observable: Observable to compute the expectation value of. If None,
            the `executor` being decorated must return an expectation value.
            Otherwise, the `QuantumResult` returned by the `executor` is used
            to compute the expectation of the observable.
        factory: Factory object determining the zero-noise extrapolation
            method.
        scale_noise: Function for scaling the noise of a quantum circuit.
        num_to_average: Number of times expectation values are computed by
            the executor after each call to scale_noise, then averaged.

    Returns:
        The error-mitigating decorator to be applied to an executor function.
    """
    # Raise an error if the decorator is used without parenthesis
    if callable(observable):
        raise TypeError(
            "Decorator must be used with parentheses (i.e., @zne_decorator()) "
            "even if no explicit arguments are passed."
        )

    def decorator(
        executor: Callable[[QPROGRAM], QuantumResult]
    ) -> Callable[[QPROGRAM], float]:
        return mitigate_executor(
            executor,
            observable,
            factory=factory,
            scale_noise=scale_noise,
            num_to_average=num_to_average,
        )

    return decorator
