from typing import List, Tuple, Callable
import mitiq.qiskit.qiskit_utils as qs_utils
from mitiq import QPROGRAM
from mitiq.factories import Factory, LinearFactory


def mitigate(pq: QPROGRAM, fac: Factory, scale_noise: Callable[[QPROGRAM, float], QPROGRAM], run_program: Callable) -> \
        Tuple[List[float], List[float]]:
    """
    Runs the factory until convergence and returns the full list of noise values and expectations that have been
    calculated.
    :param pq: Program to mitigate.
    """
    if not fac.is_converged(fac.instack, fac.outstack):
        next_param = fac.step(fac.instack, fac.outstack)

        scaled_pq = scale_noise(pq, next_param)

        next_result = run_program(scaled_pq)
        fac.outstack.append(next_result)

        return mitigate(pq, fac, scale_noise, run_program)

    return fac.instack, fac.outstack


def zne(run_program: Callable[[QPROGRAM, int], float], fac: Factory = None,
        scale_noise: Callable[[QPROGRAM, float], QPROGRAM] = None) -> Callable[[QPROGRAM], float]:
    
    if scale_noise is None:
        # TODO this assumes is qiskit
        scale_noise = qs_utils.scale_noise

    if fac is None:
        fac = LinearFactory([1.0, 1.5, 2.0])

    def zne_run(pq: QPROGRAM) -> float:
        instack, outstack = mitigate(pq, fac, scale_noise, run_program)
        return fac.reduce(instack, outstack)

    return zne_run


def zne_factory(fac: Factory = None, scale_noise: Callable[[QPROGRAM, float], QPROGRAM] = None) -> Callable:
    # more precisely, this is a wrap function which is necessary to pass parameters to the decorator
    # formally, the function below is the actual decorator
    def decorator(fun):
        return zne(fun, fac, scale_noise)

    return decorator
