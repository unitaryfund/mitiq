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
from typing import Callable, Optional
from functools import wraps

from mitiq._typing import QPROGRAM
from mitiq.zne.inference import Factory, RichardsonFactory
from mitiq.zne.scaling import fold_gates_at_random


def execute_with_zne(
    qp: QPROGRAM,
    executor: Callable[[QPROGRAM], float],
    factory: Optional[Factory] = None,
    scale_noise: Callable[[QPROGRAM, float], QPROGRAM] = fold_gates_at_random,
    num_to_average: int = 1,
) -> float:
    """Returns the zero-noise extrapolated expectation value that is computed
    by running the quantum program `qp` with the executor function.

    Args:
        qp: Quantum program to execute with error mitigation.
        executor: Executes a circuit and returns an expectation value.
        factory: Factory object that determines the zero-noise extrapolation
            method.
        scale_noise: Function for scaling the noise of a quantum circuit.
        num_to_average: Number of times expectation values are computed by
            the executor after each call to scale_noise, then averaged.
    """
    if not factory:
        factory = RichardsonFactory(scale_factors=[1.0, 2.0, 3.0])

    if not callable(executor):
        raise TypeError("Argument `executor` must be callable.")

    if not isinstance(factory, Factory):
        raise TypeError(
            f"Argument `factory` must be of type mitiq.factories.Factory "
            f"but type(factory) is {type(factory)}."
        )

    if not callable(scale_noise):
        raise TypeError("Argument `scale_noise` must be callable.")

    if num_to_average < 1:
        raise ValueError("Argument `num_to_average` must be a positive int.")

    return factory.run(qp, executor, scale_noise, int(num_to_average)).reduce()


def mitigate_executor(
    executor: Callable[[QPROGRAM], float],
    factory: Optional[Factory] = None,
    scale_noise: Callable[[QPROGRAM, float], QPROGRAM] = fold_gates_at_random,
    num_to_average: int = 1,
) -> Callable[[QPROGRAM], float]:
    """Returns an error-mitigated version of the input `executor`.

    The input `executor` executes a circuit with an arbitrary backend and
    produces an expectation value (without any error mitigation). The returned
    executor executes the circuit with the same backend but uses zero-noise
    extrapolation to produce a mitigated expectation value.

    Args:
        executor: Executes a circuit and returns an expectation value.
        factory: Factory object determining the zero-noise extrapolation
            method.
        scale_noise: Function for scaling the noise of a quantum circuit.
        num_to_average: Number of times expectation values are computed by
            the executor after each call to scale_noise, then averaged.
    """

    @wraps(executor)
    def new_executor(qp: QPROGRAM) -> float:
        return execute_with_zne(
            qp, executor, factory, scale_noise, num_to_average
        )

    return new_executor


def zne_decorator(
    factory: Optional[Factory] = None,
    scale_noise: Callable[[QPROGRAM, float], QPROGRAM] = fold_gates_at_random,
    num_to_average: int = 1,
) -> Callable[[Callable[[QPROGRAM], float]], Callable[[QPROGRAM], float]]:
    """Decorator which adds error mitigation to an executor function, i.e., a
    function which executes a quantum circuit with an arbitrary backend and
    returns an expectation value.

    Args:
        factory: Factory object determining the zero-noise extrapolation
            method.
        scale_noise: Function for scaling the noise of a quantum circuit.
        num_to_average: Number of times expectation values are computed by
            the executor after each call to scale_noise, then averaged.
    """
    # Raise an error if the decorator is used without parenthesis
    if callable(factory):
        raise TypeError(
            "Decorator must be used with parentheses (i.e., @zne_decorator()) "
            "even if no explicit arguments are passed."
        )

    def decorator(
        executor: Callable[[QPROGRAM], float]
    ) -> Callable[[QPROGRAM], float]:
        return mitigate_executor(
            executor, factory, scale_noise, num_to_average
        )

    return decorator
