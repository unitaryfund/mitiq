from pyquil import Program
<<<<<<< HEAD
from typing import List, Tuple

from mitiq.zne import richardson_extr
import mitiq.qiskit.qiskit_utils as qs_utils


class Generator(object):
    """
    This object adaptively generates new noise scaling parameters based on a historical stack of previous noise
    scale parameters and previously calculated expectation values. Noise scaling parameters are stored on the `in_stack`
    and the running list of expectation values are stored on the `out_stack`.
    """
    def __init__(self, instack: List[float], outstack: List[float]):
        self.instack = instack
        self.outstack = outstack

    def step(self, instack: List[float], outstack: List[float]):
        raise NotImplementedError

    def is_converged(self, instack: List[float], outstack: List[float]):
        raise NotImplementedError


class BatchedGenerator(Generator):
    def __init__(self, scalars: List[float], instack: List[float] = None, outstack: List[float] = None):
=======
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
>>>>>>> origin/master
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
        super(self.__class__, self).__init__(instack, outstack)
        self.scalars = scalars

<<<<<<< HEAD
    def step(self, instack: List[float], outstack: List[float]):
        next_param = self.scalars[len(outstack)]
        self.instack.append(next_param)
        return next_param

    def is_converged(self, instack: List[float], outstack: List[float]):
        return len(instack) == len(self.scalars)


class Mitigator(object):
    """
    This object matches a Generator against a backend interface by including a 'run_program' function that maps quantum
    circuits / programs to expectation values.
    """
    def __init__(self, gen: Generator, run_program):
        self.gen = gen
        self.run_program = run_program

    def mitigate(self, pq: Program, scale_noise) -> Tuple[List[float], List[float]]:
        """
        Runs the generator until convergence and return the full list of noise values and expectations that have been
        calculated.
        :param pq: Program to mitigate.
        """
        gen = self.gen
        if gen.is_converged(gen.instack, gen.outstack):
            return gen.instack, gen.outstack
        else:
            next_param = gen.step(gen.instack, gen.outstack)
            scaled_pq = scale_noise(pq, next_param)
            yy = self.run_program(scaled_pq)
            gen.outstack.append(yy)
            return self.mitigate(pq, scale_noise)


def reduce(expectations: List[float]) -> float:
    expt = richardson_extr(expectations, circuit=None, order=len(expectations)-1, c=None)
    return expt


def zne(run_program, gen=None, scale_noise=None):
=======
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
>>>>>>> origin/master
    if scale_noise is None:
        # TODO this assumes is qiskit
        scale_noise = qs_utils.scale_noise

<<<<<<< HEAD
    if gen is None:
        gen = BatchedGenerator([1.0, 2.0, 3.0])

    def zne_run(pq):
        mitigator = Mitigator(gen, run_program)
        params, expects = mitigator.mitigate(pq, scale_noise)
        return reduce(expects)
=======
    if fac is None:
        fac = BatchedFactory([1.0, 2.0, 3.0])

    def zne_run(pq: QPROGRAM) -> float:
        params, expects = mitigate(pq, fac, scale_noise, run_program)
        return fac.reduce(expects)
>>>>>>> origin/master

    return zne_run


<<<<<<< HEAD
def zne_factory(gen=None, scale_noise=None):
    # more precisely, this is a wrap function which is necessary to pass parameters to the decorator
    # formally, the function below is the actual decorator
    def decorator(fun):
        return zne(fun, gen, scale_noise)
=======
def zne_factory(fac: Factory = None, scale_noise: Callable[[QPROGRAM, float], QPROGRAM] = None) -> Callable:
    # more precisely, this is a wrap function which is necessary to pass parameters to the decorator
    # formally, the function below is the actual decorator
    def decorator(fun):
        return zne(fun, fac, scale_noise)
>>>>>>> origin/master

    return decorator
