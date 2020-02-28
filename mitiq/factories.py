from pyquil import Program
from typing import List, Tuple, Callable

import mitiq.qiskit.qiskit_utils as qs_utils
import numpy as np

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

    def reduce(self) -> float:
        raise NotImplementedError


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

# Specific extrapolation algorithms are given below.
# They are implemented as child classes of Factory or BatchedFactory.

# TODO list:
# [x] Richardson's extrapolation
# [x] Linear fit
# [x] Polynomial fit
# Log-Richardson's extrapolation
# Log-Polynomial fit
# Adaptive Linear fit
# Adaptive Log-polinomial fit
# Adaptive max-likelihood
# Adaptive Bayesian
# etc... 


class RichardsonFactory(BatchedFactory):
    """Factory object implementing the Richardson's extrapolation algorithm."""

    @staticmethod
    def get_gammas(c: float) -> List[float]:
        """Returns the linear combination coefficients "gammas" for Richardson's extrapolation.
        The input is a list of the noise stretch factors.
        """
        order = len(c) - 1
        np_c = np.asarray(c)
        A = np.zeros((order + 1, order + 1))
        for k in range(order + 1):
            A[k] = np_c ** k
        b = np.zeros(order + 1)
        b[0] = 1
        return np.linalg.solve(A, b)

    def reduce(self, x: List[float], y: List[float]) -> float:
        """ Given two lists of x and y values associated to an unknwn function y=f(x), returns 
        the extrapolation of the function to the x=0 limit, i.e., an estimate of f(0).
        The algorithm is based on the Richardson's extrapolation method.
        """

        # check arguments are valid
        assert len(y) > 0
        assert len(x) == len(y)
        # Richardson's extrapolation
        gammas = self.get_gammas(x)
        return np.dot(gammas, y)


class LinearFactory(BatchedFactory):
    """Factory object implementing a zero-noise extrapolation algotrithm based on a linear fit."""

    def reduce(self, x: List[float], y: List[float]) -> float:
        """ Given two lists of x and y values associated to an unknwn function y=f(x), returns 
        the extrapolation of the function to the x=0 limit, i.e., an estimate of f(0).
        The algorithm determines, with a standard least squared method, the parameters 
        (q, m) such that the line g(x) = q + m*x optimally fits the input points.
        Returns g(0) = q.
        """
        # linear least squared fit
        _ , q = np.polyfit(x, y, deg=1)
        return q


class PolyFactory(BatchedFactory):
    """Factory object implementing a zero-noise extrapolation algotrithm based on a polynomial fit."""

    def reduce(self, x: List[float], y: List[float], order: int) -> float:
        """ Given two lists of x and y values associated to an unknwn function y=f(x), returns 
        the extrapolation of the function to the x=0 limit, i.e., an estimate of f(0).
        The algorithm determines with a least squared method, the polynomial of degree='order' 
        which optimally fits the input points. The value of the polynomial at x=0 is returned.

            Note: order=1 corresponds to a standard linear least square fitting. 
                  In this case (q, m) are determined such that g(x) = q + m*x optimally fits 
                  data and g(0) = q is returned.
        """
        coefficients = np.polyfit(x, y, deg=order)
        return coefficients[-1]
