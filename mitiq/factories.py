"""Contains all the main classes corresponding to different zero-noise
extrapolation methods.
"""

from typing import List, Iterable, Union, Tuple
import numpy as np
from scipy.optimize import curve_fit


class Factory:
    """
    Abstract class designed to adaptively produce a new noise scaling parameter
    based on a historical stack of previous noise scale parameters
    ("self.instack") and previously estimated expectation values
    ("self.outstack").

    Specific zero-noise extrapolation algorithms, adaptive or non-adaptive,
    are derived from this class.
    A Factory object is not supposed to directly perform any quantum
    computation, only the classical results of quantum experiments are
    processed by it.
    """

    def __init__(self) -> None:
        """
        Initialization arguments (e.g. noise scale factors) depend on the
        particular extrapolation algorithm and can be added to the "__init__"
        method of the associated derived class.
        """
        self.instack = []
        self.outstack = []

    def push(self, instack_val: float, outstack_val: float) -> None:
        """
        Appends "instack_val" to "self.instack" and "outstack_val" to
        "self.outstack".
        Each time a new expectation value is computed this method should be
        used to update the internal state of the Factory.
        """
        self.instack.append(instack_val)
        self.outstack.append(outstack_val)

    def next(self) -> float:
        """Returns the next noise level to execute a circuit at."""
        raise NotImplementedError

    def is_converged(self) -> bool:
        """Returns True if all needed expectation values have been computed,
        else False.
        """
        raise NotImplementedError

    def reduce(self) -> float:
        """Returns the extrapolation to the zero-noise limit."""
        raise NotImplementedError

    def reset(self) -> None:
        """Resets the instack and outstack of the Factory to empty values."""
        self.instack = []
        self.outstack = []


class BatchedFactory(Factory):
    """
    Abstract class of a non-adaptive Factory.

    This is initialized with a given batch of scaling factors ("scalars").
    The "self.next" method trivially iterates over the elements of "scalars"
    in a non-adaptive way.
    Convergence is achieved when all the correpsonding expectation values have
    been measured.

    Specific (non-adaptive) zero-noise extrapolation algorithms can be derived
    from this class by overriding the "self.reduce" and (if necessary)
    the "__init__" method.
    """

    def __init__(self, scalars: Iterable[float]) -> None:
        """
        Args:
            scalars: Iterable of noise scale factors at which expectation
            values should be measured.
        """
        if len(scalars) == 0:
            raise ValueError(
                "The argument 'scalars' should contain at least one element."
                "At least 2 elements are necessary"\
                " for non-trivial extrapolation."
            )
        self.scalars = scalars
        super(BatchedFactory, self).__init__()

    def next(self) -> float:
        try:
            next_param = self.scalars[len(self.outstack)]
        except IndexError:
            raise IndexError(
                "BatchedFactory cannot take another step. "
                f"Number of batched scalars ({len(self.scalars)}) exceeded."
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
    Factory object implementing a zero-noise extrapolation algorithm based on
    a polynomial fit.

    Note:
        RichardsonFactory and LinearFactory are special cases of PolyFactory.

    """

    def __init__(self, scalars: Iterable[float], order: int) -> None:
        """
        Args:
            scalars: Iterable of noise scale factors at which expectation
                     values should be measured.
            order: Polynomial extrapolation order.
                   It cannot exceed len(scalars) - 1.
        """
        if order > len(scalars) - 1:
            raise ValueError(
                "The extrapolation order cannot exceed len(scalars) - 1."
            )
        self.order = order
        super(PolyFactory, self).__init__(scalars)

    @staticmethod
    def static_reduce(
        instack: List[float], outstack: List[float], order: int
    ) -> float:
        """
        Determines with a least squared method, the polynomial of degree equal
        to 'order' which optimally fits the input data.
        The zero-noise limit is returned.

        This static method is equivalent to the "self.reduce" method of
        PolyFactory, but can be called also by other factories which are
        particular cases of PolyFactory, e.g., LinearFactory
        and RichardsonFactory.
        """
        # Check arguments
        error_str = (
            "Data is not enough: at least two data points are necessary."
        )
        if instack is None or outstack is None:
            raise ValueError(error_str)
        if len(instack) != len(outstack) or len(instack) < 2:
            raise ValueError(error_str)
        if order > len(instack) - 1:
            raise ValueError(
                "Extrapolation order is too high. "
                "The order cannot exceed the number of data points minus 1."
            )
        # Get coefficients {c_j} of p(x)= c_0 + c_1*x + c_2*x**2...
        # which best fits the data
        coefficients = np.polyfit(instack, outstack, deg=order)
        # c_0, i.e., the value of p(x) at x=0, is returned
        return coefficients[-1]

    def reduce(self) -> float:
        """
        Determines with a least squared method, the polynomial of degree equal
        to "self.order" which optimally fits the input data.
        The zero-noise limit is returned.
        """
        return PolyFactory.static_reduce(
            self.instack, self.outstack, self.order
        )


class RichardsonFactory(BatchedFactory):
    """Factory object implementing Richardson's extrapolation."""

    def reduce(self) -> float:
        """Returns the Richardson's extrapolation to the zero-noise limit."""
        # Richardson's extrapolation is a particular case of a polynomial fit
        # with order equal to the number of data points minus 1.
        order = len(self.instack) - 1
        return PolyFactory.static_reduce(
            self.instack, self.outstack, order=order
        )


class LinearFactory(BatchedFactory):
    """Factory object implementing a zero-noise extrapolation algorithm based
    on a linear fit.
    """

    def reduce(self) -> float:
        """
        Determines, with a least squared method, the line of best fit
        associated to the data points. The intercept is returned.
        """
        # Richardson's extrapolation is a particular case of a polynomial fit
        # with order equal to 1.
        return PolyFactory.static_reduce(self.instack, self.outstack, order=1)


class ExpFactory(BatchedFactory):
    """
    Factory object implementing a zero-noise extrapolation algorithm assuming
    an exponential ansatz y(x) = a + b * exp(-c * x), with c > 0.

    If the asymptotic value (y(x->inf) = a) is known, a linear fit with respect
    to z(x) := log[sing(b) (y(x) - a)] is used.
    Otherwise, a non-linear fit of y(x) is performed.
    """

    def __init__(
        self, scalars: Iterable[float], asymptote: Union[float, None] = None
    ) -> None:
        """
        Args:
            scalars: Iterable of noise scale factors at which expectation
            values should be measured.
            asymptote: Infinite-noise limit (optional argument).
        """
        super(ExpFactory, self).__init__(scalars)
        if not (asymptote is None or isinstance(asymptote, float)):
            raise ValueError(
                "The argument 'asymptote' must be either a float or None"
            )
        self.asymptote = asymptote

    def reduce(self) -> float:
        """Returns the zero-noise limit, assuming an exponential ansatz:
        y(x) = a + b * exp(-c * x), with c > 0.
        """
        return PolyExpFactory.static_reduce(
            self.instack, self.outstack, self.asymptote, order=1,
        )[0]


class PolyExpFactory(BatchedFactory):
    """
    Factory object implementing a zero-noise extrapolation algorithm assuming
    an (almost) exponential ansatz with a non linear exponent, i.e.:

    y(x) = a + s * exp(z(x)), where z(x) is a polynomial of a given order.

    The parameter "s" is a sign variable which can be either 1 or -1,
    corresponding to decreasing and increasing exponentials, respectively.
    The parameter "s" is automatically deduced from the data.

    If the asymptotic value (y(x->inf) = a) is known, a linear fit with respect
    to z(x) := log[s(y(x) - a)] is used.
    Otherwise, a non-linear fit of y(x) is perfomed.
    """

    def __init__(
        self,
        scalars: Iterable[float],
        order: int,
        asymptote: Union[float, None] = None,
    ) -> None:
        """
        Args:
            scalars: Iterable of noise scale factors at which expectation
            values should be measured.
            order: Polynomial extrapolation order. Must be <=len(scalars) - 1.
                   If asymptote is None, order cannot exceed len(scalars) - 2.
            asymptote: Infinite-noise limit (optional argument).
        """
        super(PolyExpFactory, self).__init__(scalars)
        if not (asymptote is None or isinstance(asymptote, float)):
            raise ValueError(
                "The argument 'asymptote' must be either a float or None"
            )
        self.order = order
        self.asymptote = asymptote

    @staticmethod
    def static_reduce(
        instack: List[float],
        outstack: List[float],
        asymptote: Union[float, None],
        order: int,
        eps: float = 1.0e-9,
    ) -> Tuple[float, List[float]]:
        """
        Determines the zero-noise limit, assuming an exponential ansatz:
        y(x) = a + s * exp(z(x)), where z(x) is a polynomial of a given order.

        The parameter "s" is a sign variable which can be either 1 or -1,
        corresponding to decreasing and increasing exponentials, respectively.
        The parameter "s" is automatically deduced from the data.

        It is also assumed that z(x-->inf)=-inf, such that y(x-->inf)-->a.

        If asymptote is None, the ansatz y(x) is fitted with a non-linear
        optimization.
        Otherwise, a linear fit with respect to
        z(x) := log(sign * (y(x) - asymptote))
        is performed.

        This static method is equivalent to the "self.reduce" method
        of PolyExpFactory, but can be called also by other factories which are
        related to PolyExpFactory, e.g., ExpFactory, AdaExpFactory.

        Args:
            instack: x data values.
            outstack: y data values.
            asymptote: y(x->inf).
            order: Extrapolation order.
            eps: Epsilon to regularize log(sign (instack - asymptote)) when
                 the argument is to close to zero or negative.

        Returns:
            (znl, params): Where "znl" is the zero-noise-limit and "params"
                           are the optimal fitting parameters.

        """
        # Shift is 0 if asymptote is given, 1 if asymptote is not given
        shift = int(asymptote is None)
        # Check arguments
        error_str = (
            "Data is not enough: at least two data points are necessary."
        )
        if instack is None or outstack is None:
            raise ValueError(error_str)
        if len(instack) != len(outstack) or len(instack) < 2:
            raise ValueError(error_str)
        if order > len(instack) - (1 + shift):
            raise ValueError(
                "Extrapolation order is too high. "
                f"The order cannot exceed the number" \
                " of data points minus {1 + shift}."
            )

        # CASE 1: asymptote is None.
        def ansatz(x: float, *coeffs: float):
            """Ansatz function of generic order."""
            # Coefficients of the polynomial to be exponentiated
            z_coeffs = coeffs[2:][::-1]
            return coeffs[0] + coeffs[1] * np.exp(x * np.polyval(z_coeffs, x))

        if asymptote is None:
            # Initial values for the parameters
            p_zero = [-j / (j + 1.0) for j in range(order + 2)]
            opt_params, _ = curve_fit(ansatz, instack, outstack, p0=p_zero)
            # The zero limit is ansatz(0)= asympt + b
            zero_limit = opt_params[0] + opt_params[1]
            return (zero_limit, opt_params)

        # CASE 2: asymptote is given.
        # deduce if the exponential is a decay or a growth
        mean_y = sum(outstack) / len(outstack)
        sign = np.sign(mean_y - asymptote)
        # Polynomial fit of z(x).
        zstack = [np.log(max(sign * (y - asymptote), eps)) for y in outstack]
        # Get coefficients {z_j} of z(x)= z_0 + z_1*x + z_2*x**2...
        # Note: coefficients are ordered from high powers of x to low powers x.
        z_coefficients = np.polyfit(instack, zstack, deg=order)
        zero_limit = asymptote + sign * np.exp(z_coefficients[-1])
        # Parameters from low order to high order
        params = [asymptote] + list(z_coefficients[::-1])
        return (zero_limit, params)

    def reduce(self) -> float:
        """Returns the zero-noise limit, assuming an exponential ansatz:
        y(x) = a + s * exp(z(x)), where z(x) is a polynomial of a given order.
        The parameter "s" is a sign variable which can be either 1 or -1,
        corresponding to decreasing and increasing exponentials, respectively.
        The parameter "s" is automatically deduced from the data.
        It is also assumed that z(x-->inf)=-inf, such that y(x-->inf)-->a.
        """

        return self.static_reduce(
            self.instack, self.outstack, self.asymptote, self.order,
        )[0]


class AdaExpFactory(Factory):
    """Factory object implementing an adaptive zero-noise extrapolation
    algorithm assuming an exponential ansatz y(x) = a + b * exp(-c * x),
    with c > 0.

    The noise scale factors are are chosen adaptively at each step,
    depending on the history of collected results.

    If the asymptotic value (y(x->inf) = a) is known, a linear fit with respect
    to z(x) := log[sing(b) (y(x) - a)] is used.
    Otherwise, a non-linear fit of y(x) is performed.
    """

    _SHIFT_FACTOR = 1.27846

    def __init__(
        self,
        steps: int,
        scalar: float = 2,
        asymptote: Union[float, None] = None,
    ) -> None:
        """Instantiate a new object of this Factory class.

        Args:
            steps: The number of optimization steps. At least 3 are necessary.
            scalar: The second noise scale factor (the first is always 1.0).
                    Further scale factors are adaptively determined.
            asymptote: The infinite noise limit (if known) of the expectation
                       value. Default is None.
        """
        super(AdaExpFactory, self).__init__()
        if not (asymptote is None or isinstance(asymptote, float)):
            raise ValueError(
                "The argument 'asymptote' must be either a float or None"
            )
        if scalar <= 1:
            raise ValueError(
                "The argument 'scalar' must be strictly larger than one."
            )
        if steps < 3 + int(asymptote is None):
            raise ValueError(
                "The argument 'steps' must be an integer"\
                " greater or equal to 3. "
                "If 'asymptote' is None, 'steps' must be"\
                " greater or equal to 4."
            )
        self.steps = steps
        self.scalar = scalar
        self.asymptote = asymptote
        # Keep a log of the optimization process storing:
        # noise value(s), expectation value(s), parameters, and zero limit
        self.history = (
            []
        )  # type: List[Tuple[List[float], List[float], List[float], float]]

    def next(self) -> float:
        """Returns the next noise level to execute a circuit at."""
        # The 1st noise scale parameter is always 1
        if len(self.instack) == 0:
            return 1.0
        # The 2nd noise scale parameter is self.scalar
        if len(self.instack) == 1:
            return self.scalar
        # If asymptote is None we use 2 * scalar as third noise parameter
        if (len(self.instack) == 2) and (self.asymptote is None):
            return 2 * self.scalar
        # Call self.reduce() in order to update self.history
        self.reduce()
        # Get the most recent fitted parameters from self.history
        _, _, params, _ = self.history[-1]
        # The exponent parameter is the 3rd element of params
        exponent = params[2]
        # Further noise scale factors are determined with
        # an adaptive rule which depends on self.exponent
        return 1.0 + self._SHIFT_FACTOR / np.abs(exponent)

    def is_converged(self) -> bool:
        """Returns True if all the needed expectation values have been
        computed, else False.
        """
        if len(self.outstack) != len(self.instack):
            raise IndexError(
                f"The length of 'self.instack' ({len(self.instack)}) "
                f"and 'self.outstack' ({len(self.outstack)}) must be equal."
            )
        return len(self.outstack) == self.steps

    def reduce(self) -> float:
        """Returns the zero-noise limit, assuming an exponential ansatz:
        y(x) = a + b * exp(-c * x), with c > 0.
        """
        zero_limit, params = PolyExpFactory.static_reduce(
            self.instack, self.outstack, self.asymptote, order=1,
        )
        # Update optimization history
        self.history.append((self.instack, self.outstack, params, zero_limit))
        return zero_limit
