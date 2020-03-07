"""Zero-noise extrapolation tools."""

from typing import Callable
import mitiq.qiskit.qiskit_utils as qs_utils
from mitiq import QPROGRAM
from mitiq.factories import Factory, LinearFactory


def run_factory(
        fac: Factory, noise_to_expval: Callable[[float], float], max_iterations: int = 100,
    ) -> None:
    """
    Runs a factory until convergence (or until the number of iterations reach "max_iterations").

    Args:
        fac: Instance of Factory object to be run.
        noise_to_expval: Function mapping noise scale values to expectation vales.
        max_iterations: Maximum number of iterations (optional). Default value is 100.
    """
    counter = 0
    while not fac.is_converged() and counter < max_iterations:
        next_param = fac.next()
        next_result = noise_to_expval(next_param)
        fac.push(next_param, next_result)
        counter += 1

    if counter == max_iterations:
        raise Warning(
            "Factory iteration loop stopped before convergence. "
            f"Maximum number of iterations ({max_iterations}) was reached."
        )

    return None

# quantum version of run_factory. Similar to the old "mitigate".
def qrun_factory(fac: Factory, qp: QPROGRAM, executor: Callable[[QPROGRAM], float], \
                 scale_noise: Callable[[QPROGRAM, float], QPROGRAM]) -> None:
    """
    Runs the factory until convergence executing quantum circuits with different noise levels.

    Args:
        fac: Factory object to run until convergence.
        qp: Circuit to mitigate.
        executor: Function which executes a circuit and returns an expectation value.
        scale_noise: Function which scales the noise level of a quantum circuit.
    """

    def _noise_to_expval(noise_param: float) -> float:
        """Evaluates the quantum expectation value for a given noise_param"""
        scaled_qp = scale_noise(qp, noise_param)

        return executor(scaled_qp)

    run_factory(fac, _noise_to_expval)


def execute_with_zne(
        qp: QPROGRAM, executor: Callable[[QPROGRAM], float], fac: Factory = None,
        scale_noise: Callable[[QPROGRAM, float], QPROGRAM] = None,
    ) -> Callable[[QPROGRAM], float]:
    """
    Takes as input a quantum circuit and returns the associated expectation value
    evaluated with error mitigation.

    Args:
        qp: Quantum circuit to execute with error mitigation.
        executor: Function executing a circuit and producing an expectation value
                  (without error mitigation).
        fac: Factory object determining the zero-noise extrapolation algorithm.
             If not specified, LinearFactory([1.0, 2.0]) will be used.
        scale_noise: Function for scaling the noise of a quantum circuit.
                     If not specified, a default method will be used.
    """
    if scale_noise is None:
        # TODO this assumes is qiskit
        scale_noise = qs_utils.scale_noise
    if fac is None:
        fac = LinearFactory([1.0, 2.0])
    qrun_factory(fac, qp, executor, scale_noise)

    return fac.reduce()


# Similar to the old "zne".
def mitigate_executor(
        executor: Callable[[QPROGRAM], float], fac: Factory = None,
        scale_noise: Callable[[QPROGRAM, float], QPROGRAM] = None,
    ) -> Callable[[QPROGRAM], float]:
    """
    Takes as input a generic function ("executor"), difined by the user, which executes a circuit
    with an arbitrary backend and produces an expectation value.

    Returns an error-mitigated version of the input "executor", having the same signature and
    automatically performing zero-noise extrapolation at each call.

    Args:
        executor: Function (to be mitigated) executing a circuit and returning an expectation value.
        fac: Factory object determining the zero-noise extrapolation algorithm.
             If not specified, LinearFactory([1.0, 2.0]) is used.
        scale_noise: Function for scaling the noise of a quantum circuit.
                     If not specified, a default method is used.
    """

    def new_executor(qp: QPROGRAM) -> float:
        return execute_with_zne(qp, executor, fac, scale_noise)

    return new_executor


def zne_decorator(
        fac: Factory = None, scale_noise: Callable[[QPROGRAM, float], QPROGRAM] = None,
    ) -> Callable[[QPROGRAM], float]:
    """
    Decorator which automatically adds error mitigation to any circuit-executor function
    defined by the user.

    It is supposed to be applied to any function which executes a quantum circuit
    with an arbitrary backend and produces an expectation value.

    Args:
        fac: Factory object determining the zero-noise extrapolation algorithm.
             If not specified, LinearFactory([1.0, 2.0]) will be used.
        scale_noise: Function for scaling the noise of a quantum circuit.
                     If not specified, a default method will be used.
    """
    # Formally, the function below is the actual decorator, while the function
    # "zne_decorator" is necessary to give additional arguments to "decorator".
    def decorator(executor: Callable[[QPROGRAM], float]) -> Callable[[QPROGRAM], float]:
        return mitigate_executor(executor, fac, scale_noise)

    return decorator
