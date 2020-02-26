"""List of zero-noise extrapolation algorithms. Each algorithm is given in the form of a Generator object."""

from typing import List, Tuple
from mitiq.adaptive_zne import BatchedFactory
# from mitiq.zne import get_gammas
import numpy as np

# TODO list:
# [x] Richardson's extrapolation
# [x] Polynomial fit
# Log-Richardson's extrapolation
# Log-Polynomial fit
# Adaptive Linear fit
# Adaptive Log-polinomial fit
# Adaptive max-likelihood
# Adaptive Bayesian
# etc... 


class RichardsonExtr(BatchedFactory):
    """Generator object implementing the Richardson's extrapolation algorithm."""

    def reduce(self, x: List[float], y: List[float]) -> float:
        """ Given two lists of x and y values associated to an unknwn function y=f(x), returns 
        the extrapolation of the function to the x=0 limit, i.e., an estimate of f(0).
        The algorithm is based on the Richardson's extrapolation method.
        """

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
        
        # check arguments are valid
        assert len(y) > 0
        assert len(x) == len(y)
        # Richardson's extrapolation
        gammas = get_gammas(x)
        return np.dot(gammas, y)


class LinearExtr(BatchedFactory):
    """Generator object implementing a zero-noise extrapolation algotrithm based on a linear fit."""

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


class PolyExtr(BatchedFactory):
    """Generator object implementing a zero-noise extrapolation algotrithm based on a polynomial fit."""

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