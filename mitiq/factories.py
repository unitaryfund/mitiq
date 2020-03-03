from typing import List, Iterable
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
    """
    Runs a series of scalar noise parameters serially.
    :param scalars: List of scalar noise values to be executed.
    :param instack: Running stack of noise values run so far.
    :param outstack: Expectations calculated thus far.
    """
    def __init__(self, scalars: Iterable[float], instack: List[float] = None, outstack: List[float] = None) -> None:
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


class RichardsonFactory(BatchedFactory):
    """Factory object implementing Richardson's extrapolation."""
    
    def reduce(self, instack: List[float], outstack: List[float]) -> float:
        """Returns the Richardson's extrapolation to the zero-noise limit."""
        # Richardson's extrapolation is a particular case of a polynomial fit
        # with order equal to the number of data points minus 1.
        order = len(x) - 1
        return  PolyFactory.static_reduce(x, y, order=order)

class LinearFactory(BatchedFactory):
    """Factory object implementing a zero-noise extrapolation algotrithm based on a linear fit."""
    
    def reduce(self, instack: List[float], outstack: List[float]) -> float:
        """
        Determines, with a least squared method, the line of best fit
        associated to the data points. The intercept is returned.
        """
        # Richardson's extrapolation is a particular case of a polynomial fit
        # with order equal to 1.
        return PolyFactory.static_reduce(x, y, order=1)


class PolyFactory(BatchedFactory):
    """
    Factory object implementing a zero-noise extrapolation algotrithm based on a polynomial fit.     
    Note: RichardsonFactory and LinearFactory are special cases of PolyFactory.
    """
    @staticmethod
    def static_reduce(x: List[float], y: List[float], order: int) -> float:
        """
        Static method equivalent to the reduce instance method of PolyFactory.
        This method is also called by other factories, e.g., LinearFactory and RichardsonFactory.
        """

        # check arguments
        error_str = "Data is not enough: at least two data points are necessary."
        if (x is None) or (y is None):
            raise ValueError(error_str) 
        if (len(x) != len(y)) or (len(x)<2):
            raise ValueError(error_str) 
        if order > len(x) - 1:
            raise ValueError(
                "Extrapolation order is too high. The order cannot exceed the number of data points minus 1."
            )
        
        # get coefficients c_j of p(x)= c_0 + c_1*x + c_2*x**2... which best fits the data
        coefficients = np.polyfit(x, y, deg=order)
        # c_0, i.e., p(x=0), is returned
        return coefficients[-1]

    def reduce(self, instack: List[float], outstack: List[float], order: int) -> float:
        """
        Determines with a least squared method, the polynomial of degree equal to 'order' 
        which optimally fits the input data. The zero-noise limit is returned.
        """
        return PolyFactory.static_reduce(x, y, order)
