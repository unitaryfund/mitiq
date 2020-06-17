"""High-level zero-noise extrapolation tools."""

from typing import Callable

from mitiq import QPROGRAM
from mitiq.factories import Factory, RichardsonFactory
from mitiq.folding import fold_gates_at_random


def execute_with_zne(
    qp: QPROGRAM,
    executor: Callable[[QPROGRAM], float],
    factory: Factory = RichardsonFactory(scale_factors=[1., 2., 3.]),
    scale_noise: Callable[[QPROGRAM, float], QPROGRAM] = fold_gates_at_random,
) -> float:
    """Returns the zero-noise extrapolated expectation value that is computed
    by running the quantum program `qp` with the executor function.

    Args:
        qp: Quantum program to execute with error mitigation.
        executor: Executes a circuit and returns an expectation value.
        factory: Factory object determining the zero-noise extrapolation method.
        scale_noise: Function for scaling the noise of a quantum circuit.
    """
    if not callable(executor):
        raise TypeError("Argument `executor` must be callable.")

    if not isinstance(factory, Factory):
        raise TypeError(
            f"Argument `factory` must be of type mitiq.factories.Factory "
            f"but type(factory) is {type(factory)}."
        )

    if not callable(scale_noise):
        raise TypeError("Argument `scale_noise` must be callable.")

    return factory.run(qp, executor, scale_noise).reduce()


def mitigate_executor(
    executor: Callable[[QPROGRAM], float],
    factory: Factory = RichardsonFactory(scale_factors=[1., 2., 3.]),
    scale_noise: Callable[[QPROGRAM, float], QPROGRAM] = fold_gates_at_random,
) -> Callable[[QPROGRAM], float]:
    """Returns an error-mitigated version of the input `executor`.

    The input `executor` executes a circuit with an arbitrary backend and
    produces an expectation value (without any error mitigation). The returned
    executor executes the circuit with the same backend but uses zero-noise
    extrapolation to produce a mitigated expectation value.

    Args:
        executor: Executes a circuit and returns an expectation value.
        factory: Factory object determining the zero-noise extrapolation method.
        scale_noise: Function for scaling the noise of a quantum circuit.
    """
    def new_executor(qp: QPROGRAM) -> float:
        return execute_with_zne(qp, executor, factory, scale_noise)
    return new_executor


def zne_decorator(
    factory: Factory = RichardsonFactory(scale_factors=[1., 2., 3.]),
    scale_noise: Callable[[QPROGRAM, float], QPROGRAM] = fold_gates_at_random,
) -> Callable[[QPROGRAM], float]:
    """Decorator which adds error mitigation to an executor function, i.e., a
    function which executes a quantum circuit with an arbitrary backend and
    returns an expectation value.

    Args:
        factory: Factory object determining the zero-noise extrapolation method.
        scale_noise: Function for scaling the noise of a quantum circuit.
    """
    def decorator(
        executor: Callable[[QPROGRAM], float]
    ) -> Callable[[QPROGRAM], float]:
        return mitigate_executor(executor, factory, scale_noise)
    return decorator
