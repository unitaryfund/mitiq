"""Contains all the main classes corresponding to different zero-noise extrapolation methods."""

from typing import List, Iterable
import numpy as np


class Factory:
    """
    Abstract class designed to adaptively produce a new noise scaling parameter based on a historical
    stack of previous noise scale parameters ("self.instack") and previously estimated expectation
    values ("self.outstack").

    Specific zero-noise extrapolation algorithms, adaptive or non-adaptive, are derived from this class.
    A Factory object is not supposed to directly perform any quantum computation, only the classical
    results of quantum experiments are processed by it.
    """

    def __init__(self) -> None:
        """
        Initialization arguments (e.g. noise scale factors) depend on the particular extrapolation
        algorithm and can be added to the "__init__" method of the associated derived class.
        """
        self.instack = []
        self.outstack = []

    def push(self, instack_val: float, outstack_val: float) -> None:
        """
        Appends "instack_val" to "self.instack" and "outstack_val" to "self.outstack".
        Each time a new expectation value is computed this method should be used
        to update the internal state of the Factory.
        """
        self.instack.append(instack_val)
        self.outstack.append(outstack_val)

    def next(self) -> float:
        """Returns the next noise level to execute a circuit at."""
        raise NotImplementedError

    def is_converged(self) -> bool:
        """Returns True if all needed expectation values have been computed, else False."""
        raise NotImplementedError

    def reduce(self) -> float:
        """Returns the extrapolation to the zero-noise limit."""
        raise NotImplementedError


class BatchedFactory(Factory):
    """
    Abstract class of a non-adaptive Factory.

    This is initialized with a given batch of scaling factors ("scalars").
    The "self.next" method trivially iterates over the elements of "scalars"
    in a non-adaptive way.
    Convergence is achieved when all the correpsonding expectation values have been measured.

    Specific (non-adaptive) zero-noise extrapolation algorithms can be derived from this class by
    overriding the "self.reduce" and (if necessary) the "__init__" method.
    """

    def __init__(self, scalars: Iterable[float]) -> None:
        """
        Args:
            scalars: Iterable of noise scale factors at which expectation values should be measured.
        """
        if len(scalars) == 0:
            raise ValueError(
                "The argument 'scalars' should contain at least one element."
                "At least 2 elements are necessary for a non-trivial extrapolation."
            )
        self.scalars = scalars
        super(BatchedFactory, self).__init__()

    def next(self) -> float:
        try:
            next_param = self.scalars[len(self.outstack)]
        except IndexError:
            raise IndexError(
                f"BatchedFactory cannot take another step. Number of batched scalars ({len(self.scalars)}) "
                "exceeded."
            )
        return next_param

    def is_converged(self) -> bool:
        if len(self.outstack) != len(self.instack):
            raise IndexError(
                f"The length of 'self.instack' ({len(self.instack)}) "
                f"and 'self.outstack' ({len(self.outstack)}) must be equal."
            )
        return len(self.outstack) == len(self.scalars)


class PolyFactory(BatchedFactory):
    """
    Factory object implementing a zero-noise extrapolation algotrithm based on a polynomial fit.
    Note: RichardsonFactory and LinearFactory are special cases of PolyFactory.
    """

    def __init__(self, scalars: Iterable[float], order: int) -> None:
        """
        Args:
            scalars: Iterable of noise scale factors at which expectation values should be measured.
            order: Polynomial extrapolation order. It cannot exceed len(scalars) - 1.
        """
        if order > len(scalars) - 1:
            raise ValueError("The extrapolation order cannot exceed len(scalars) - 1.")
        self.order = order
        super(PolyFactory, self).__init__(scalars)

    @staticmethod
    def static_reduce(instack: List[float], outstack: List[float], order: int) -> float:
        """
        Determines with a least squared method, the polynomial of degree equal to 'order'
        which optimally fits the input data. The zero-noise limit is returned.

        This static method is equivalent to the "self.reduce" method of PolyFactory, but
        can be called also by other factories which are particular cases of PolyFactory,
        e.g., LinearFactory and RichardsonFactory.
        """
        # check arguments
        error_str = "Data is not enough: at least two data points are necessary."
        if instack is None or outstack is None:
            raise ValueError(error_str)
        if len(instack) != len(outstack) or len(instack) < 2:
            raise ValueError(error_str)
        if order > len(instack) - 1:
            raise ValueError(
                "Extrapolation order is too high. The order cannot exceed the number of data points minus 1."
            )
        # get coefficients {c_j} of p(x)= c_0 + c_1*x + c_2*x**2... which best fits the data
        coefficients = np.polyfit(instack, outstack, deg=order)
        # c_0, i.e., the value of p(x) at x=0, is returned
        return coefficients[-1]

    def reduce(self) -> float:
        """
        Determines with a least squared method, the polynomial of degree equal to "self.order"
        which optimally fits the input data. The zero-noise limit is returned.
        """
        return PolyFactory.static_reduce(self.instack, self.outstack, self.order)


class RichardsonFactory(BatchedFactory):
    """Factory object implementing Richardson's extrapolation."""

    def reduce(self) -> float:
        """Returns the Richardson's extrapolation to the zero-noise limit."""
        # Richardson's extrapolation is a particular case of a polynomial fit
        # with order equal to the number of data points minus 1.
        order = len(self.instack) - 1
        return PolyFactory.static_reduce(self.instack, self.outstack, order=order)


class LinearFactory(BatchedFactory):
    """Factory object implementing a zero-noise extrapolation algotrithm based on a linear fit."""

    def reduce(self) -> float:
        """
        Determines, with a least squared method, the line of best fit
        associated to the data points. The intercept is returned.
        """
        # Richardson's extrapolation is a particular case of a polynomial fit
        # with order equal to 1.
        return PolyFactory.static_reduce(self.instack, self.outstack, order=1)
