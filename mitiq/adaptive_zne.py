from pyquil import Program
from typing import List, Tuple, Callable

from mitiq.zne import richardson_extr
import mitiq.qiskit.qiskit_utils as qs_utils
from mitiq import QPROGRAM


class Factory(object):
    """
    This object adaptively produces new noise scaling parameters based on a historical stack of previous noise
    scale parameters and previously calculated expectation values. Noise scaling parameters are stored on the `in_stack`
    and the running list of expectation values are stored on the `out_stack`.
    """
    def __init__(self, instack: List[float], outstack: List[float]) -> None:
        self.instack = instack
        self.outstack = outstack

    def step(self, instack: List[float], outstack: List[float]) -> None:
        raise NotImplementedError

    def is_converged(self, instack: List[float], outstack: List[float]) -> None:
        raise NotImplementedError

    def reduce(self, expectations: List[float]) -> float:
        expt = richardson_extr(expectations, circuit=None, order=len(expectations)-1, c=None)
        return expt


class BatchedFactory(Factory):
    def __init__(self, scalars: List[float], instack: List[float] = None, outstack: List[float] = None) -> None:
        """
        Runs a series of scalar noise parameters serially.
        :param scalars: List of scalar noise values to be executed.
        :param instack: Running stack of noise values run so far.
        :param outstack: Expectations calculated thus far.
        """
        if instack is None:
            instack = []
        if outstack is None:
            outstack = []
        super(BatchedFactory, self).__init__(instack, outstack)
        self.scalars = scalars

    def step(self, instack: List[float], outstack: List[float]) -> float:
        try:
            next_param = self.scalars[len(outstack)]
        except IndexError:
            raise IndexError(f"BatchedFactory cannot take another step. Number of batched scalars ({len(instack)}) "
                             f"exceeded.")
        self.instack.append(next_param)
        return next_param

    def is_converged(self, instack: List[float], outstack: List[float]) -> bool:
        return len(instack) == len(self.scalars)


def mitigate(pq: Program, fac: Factory, scale_noise: Callable, run_program: Callable) -> \
        Tuple[List[float], List[float]]:
    """
    Runs the factory until convergence and return the full list of noise values and expectations that have been
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
        fac = BatchedFactory([1.0, 2.0, 3.0])

    def zne_run(pq: QPROGRAM) -> float:
        params, expects = mitigate(pq, fac, scale_noise, run_program)
        return fac.reduce(expects)

    return zne_run


def zne_factory(fac: Factory = None, scale_noise: Callable[[QPROGRAM, float], QPROGRAM] = None) -> Callable:
    # more precisely, this is a wrap function which is necessary to pass parameters to the decorator
    # formally, the function below is the actual decorator
    def decorator(fun):
        return zne(fun, fac, scale_noise)

    return decorator
