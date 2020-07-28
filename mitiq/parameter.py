# parameter.py

from typing import Callable
import numpy as np
from cirq import Circuit

from mitiq.factories import Factory
from mitiq import QPROGRAM


def Parameter(fac: Factory) -> Factory:
    """
    Adapt a Factory to be a Parameter scaling Factory. For example:
        param_linear = Parameter(LinearFactory([1,2,3])
        param_exp = Parameter(ExpFactory([1,2,3]))

    :param fac: The Factory to adapt
    :return: The adapted Factory
    """
    fac.run = parameter_run
    return fac


def parameter_run(self, qp: QPROGRAM,
                  executor: Callable[[QPROGRAM], float],
                  apply_noise: Callable[[QPROGRAM, float], QPROGRAM],
                  shots: int,
                  max_iterations: int = 100) -> Factory:
    """
    Runs the factory until convergence executing quantum circuits.
    Accepts different noise levels.

    Args:
        qp: Circuit to mitigate.
        executor: Function executing a circuit; returns an expectation
                  value.
        apply_noise: Function applying random tweaks to parameter values in
                  order to scale the noise.
        shots: The number of shots to take for each expectation value.
        max_iterations: Maximum number of iterations (optional). Default: 100.
    """
    def _noise_to_expval(noise_param: float) -> float:
        """Evaluates the quantum expectation value for a given noise_
        param"""
        outcome = []
        for _ in range(shots):
            qp = apply_noise(qp, noise_param)
            outcome.append(executor(qp))
        return np.mean(outcome)

    return self.iterate(_noise_to_expval, max_iterations)


def param_scale_noise(qc: QPROGRAM, scale_factor: float) -> QPROGRAM:
    """ Alters the parameters in the program for a given scale_factor """
    # TODO
    return Circuit()
