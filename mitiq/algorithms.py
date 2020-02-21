"""List of zero-noise extrapolation algorithms. Each algorithm is given in the form of a Generator object."""

from typing import List, Tuple
from mitiq.adaptive_zne import BatchedGenerator
# from mitiq.zne import get_gammas
import numpy as np

# List of algorithms that we could implememnt:
# Richardson's extrapolation
# Polynomial fit
# Log-Richardson's extrapolation
# Log-Polynomial fit
# Adaptive Linear fit
# Adaptive Log-polinomial fit
# Adaptive max-likelihood
# Adaptive Bayesian
# etc... 


class RichardsonExtr(BatchedGenerator):
    """Generator object implementing the Richardson's extrapolation algorithm."""

    @staticmethod
    def reduce(x: List[float], y: List[float]) -> float:
        """ Given two lists of x and y values associated to an unknwn function y=f(x), returns 
        the extrapolation of the function to the x=0 limit, i.e., an estimate of f(0).
        The algorithm is based on the Richardson's extrapolation method.
        """

        # This function is placed here in order to make the algorithm fully self-contained.
        # I don't know if this is good or bad.
        # Alternatively, we can uncomment the line "from mitiq.zne import get_gammas" above.
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



class LinearExtr(BatchedGenerator):
    """Generator object implementing a zero-noise extrapolation algotrithm based on a linear fit."""

    @staticmethod
    def reduce(x: List[float], y: List[float]) -> float:
        """ Given two lists of x and y values associated to an unknwn function y=f(x), returns 
        the extrapolation of the function to the x=0 limit, i.e., an estimate of f(0).
        The algorithm determines, with a standard least squared method, the parameters 
        (a, b) such that the line g(x) = q + m*x optimally fits the input points.
        Returns g(0) = q.
        """
        # linear least squared fit
        m, q = np.polyfit(x, y, deg=1)
        return q