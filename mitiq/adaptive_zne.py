from pyquil import Program
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

    def reduce(self, expectations: List[float]) -> float:
        expt = richardson_extr(expectations, circuit=None, order=len(expectations)-1, c=None)
        return expt


class BatchedGenerator(Generator):
    def __init__(self, scalars: List[float], instack: List[float] = None, outstack: List[float] = None):
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


def zne(run_program, gen=None, scale_noise=None):
    if scale_noise is None:
        # TODO this assumes is qiskit
        scale_noise = qs_utils.scale_noise

    if gen is None:
        gen = BatchedGenerator([1.0, 2.0, 3.0])

    def zne_run(pq):
        mitigator = Mitigator(gen, run_program)
        params, expects = mitigator.mitigate(pq, scale_noise)
        return mitigator.gen.reduce(expects)

    return zne_run


def zne_factory(gen=None, scale_noise=None):
    # more precisely, this is a wrap function which is necessary to pass parameters to the decorator
    # formally, the function below is the actual decorator
    def decorator(fun):
        return zne(fun, gen, scale_noise)

    return decorator
