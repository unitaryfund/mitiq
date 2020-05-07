"""Zero-noise extrapolation tools."""

from typing import Callable

from mitiq import QPROGRAM
from mitiq.factories import Factory, RichardsonFactory
from mitiq.folding import fold_gates_at_random


def execute_with_zne(
    qp: QPROGRAM,
    executor: Callable[[QPROGRAM], float],
    fac: Factory = None,
    scale_noise: Callable[[QPROGRAM, float], QPROGRAM] = None,
) -> float:
    """
    Takes as input a quantum circuit and returns the associated expectation
    value evaluated with error mitigation.

    Args:
        qp: Quantum circuit to execute with error mitigation.
        executor: Function executing a circuit and producing an expect. value
                  (without error mitigation).
        fac: Factory object determining the zero-noise extrapolation algorithm.
             If not specified, LinearFactory([1.0, 2.0]) will be used.
        scale_noise: Function for scaling the noise of a quantum circuit.
                     If not specified, a default method will be used.
    """
    if scale_noise is None:
        scale_noise = fold_gates_at_random
    if fac is None:
        fac = RichardsonFactory([1.0, 2.0, 3.0])
    fac.run(qp, executor, scale_noise)

    return fac.reduce()


# Similar to the old "zne".
def mitigate_executor(
    executor: Callable[[QPROGRAM], float],
    fac: Factory = None,
    scale_noise: Callable[[QPROGRAM, float], QPROGRAM] = None,
) -> Callable[[QPROGRAM], float]:
    """
    Returns an error-mitigated version of the input "executor".
    Takes as input a generic function ("executor"), defined by the user,
    that executes a circuit with an arbitrary backend and produces an
    expectation value.

    Returns an error-mitigated version of the input "executor",
    having the same signature and automatically performing ZNE at each call.

    Args:
        executor: Function executing a circuit and returning an exp. value.
        fac: Factory object determining the zero-noise extrapolation algorithm.
             If not specified, LinearFactory([1.0, 2.0]) is used.
        scale_noise: Function for scaling the noise of a quantum circuit.
                     If not specified, a default method is used.
    """

    def new_executor(qp: QPROGRAM) -> float:
        return execute_with_zne(qp, executor, fac, scale_noise)

    return new_executor


def zne_decorator(
    fac: Factory = None,
    scale_noise: Callable[[QPROGRAM, float], QPROGRAM] = None,
) -> Callable[[QPROGRAM], float]:
    """
    Decorator which automatically adds error mitigation to any circuit-executor
     function defined by the user.

    It is supposed to be applied to any function which executes a quantum
     circuit with an arbitrary backend and produces an expectation value.

    Args:
        fac: Factory object determining the zero-noise extrapolation algorithm.
             If not specified, LinearFactory([1.0, 2.0]) will be used.
        scale_noise: Function for scaling the noise of a quantum circuit.
                     If not specified, a default method will be used.
    """
    # Formally, the function below is the actual decorator, while the function
    # "zne_decorator" is necessary to give additional arguments to "decorator".
    def decorator(
        executor: Callable[[QPROGRAM], float]
    ) -> Callable[[QPROGRAM], float]:
        return mitigate_executor(executor, fac, scale_noise)

    return decorator
