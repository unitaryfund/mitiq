"""Contains all the main classes corresponding to different zero-noise
extrapolation methods.
"""

import warnings
from mitiq import QPROGRAM
from typing import List, Iterable, Optional, Tuple, Callable
from abc import ABC, abstractmethod
import numpy as np
from numpy.lib.polynomial import RankWarning
from scipy.optimize import curve_fit, OptimizeWarning


class ExtrapolationError(Exception):
    """Error raised by :class:`.Factory` objects when
    the extrapolation fit fails.
    """
    pass


_EXTR_ERR = ("The extrapolation fit failed to converge."
             " The problem may be solved by switching to a more stable"
             " extrapolation model such as `LinearFactory`.")


class ExtrapolationWarning(Warning):
    """Warning raised by :class:`.Factory` objects when
    the extrapolation fit is ill-conditioned.
    """
    pass


_EXTR_WARN = (" The extrapolation fit may be ill-conditioned."
              " Likely, more data points are necessary to fit the parameters"
              " of the model.")


class ConvergenceWarning(Warning):
    """Warning raised by :class:`.Factory` objects when
    their `iterate` method fails to converge.
    """
    pass


def _mitiq_curve_fit(ansatz: Callable[..., float],
                     instack: List[float],
                     outstack: List[float],
                     init_params: Optional[List[float]] = None,
                     ) -> List[float]:
    """This is a wrapping of the `scipy.optimize.curve_fit` function with
    custom errors and warnings. It is used to make a non-linear fit.

    Args:
        ansatz : The model function used for zero-noise extrapolation.
                 The first argument is the noise scale variable,
                 the remaining arguments are the parameters to fit.
        instack: The array of noise scale factors.
        outstack: The array of expectation values.
        init_params: Initial guess for the parameters.
                     If None, the initial values are set to 1.
    Returns:
        opt_params: The array of optimal parameters.
    Raises:
        ExtrapolationError: If the extrapolation fit fails.
        ExtrapolationWarning: If the extrapolation fit is ill-conditioned.
    """
    try:
        with warnings.catch_warnings(record=True) as warn_list:
            opt_params, _ = curve_fit(ansatz,
                                      instack,
                                      outstack,
                                      p0=init_params)
        for warn in warn_list:
            # replace OptimizeWarning with ExtrapolationWarning
            if warn.category is OptimizeWarning:
                warn.category = ExtrapolationWarning
                warn.message = _EXTR_WARN
            # re-raise all warnings
            warnings.warn(warn.message, warn.category)
    except RuntimeError:
        raise ExtrapolationError(_EXTR_ERR) from None
    return opt_params


def _mitiq_polyfit(instack: List[float],
                   outstack: List[float],
                   deg: int,
                   weights: Optional[List[float]] = None) -> List[float]:
    """This is a wrapping of the `numpy.polyfit` function with
    custom warnings. It is used to make a polynomial fit.

    Args:
        instack: The array of noise scale factors.
        outstack: The array of expectation values.
        deg: The degree of the polynomial fit.
        weights: Optional array of weights for each sampled point.
                 This is used to make a weighted least squares fit.
    Returns:
        opt_params: The array of optimal parameters.
    Raises:
        ExtrapolationWarning: If the extrapolation fit is ill-conditioned.
    """
    with warnings.catch_warnings(record=True) as warn_list:
        opt_params = np.polyfit(instack, outstack, deg, w=weights)
    for warn in warn_list:
        # replace RankWarning with ExtrapolationWarning
        if warn.category is RankWarning:
            warn.category = ExtrapolationWarning
            warn.message = _EXTR_WARN
        # re-raise all warnings
        warnings.warn(warn.message, warn.category)
    return opt_params


class Factory(ABC):
    """
    Abstract class designed to adaptively produce a new noise scaling
    parameter based on a historical stack of previous noise scale parameters
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

    @abstractmethod
    def next(self) -> float:
        """Returns the next noise level to execute a circuit at."""
        raise NotImplementedError

    @abstractmethod
    def is_converged(self) -> bool:
        """Returns True if all needed expectation values have been computed,
        else False.
        """
        raise NotImplementedError

    @abstractmethod
    def reduce(self) -> float:
        """Returns the extrapolation to the zero-noise limit."""
        raise NotImplementedError

    def reset(self) -> None:
        """Resets the instack and outstack of the Factory to empty values."""
        self.instack = []
        self.outstack = []

    def iterate(self, noise_to_expval: Callable[[float], float],
                max_iterations: int = 100) -> 'Factory':
        """
        Runs a factory until convergence (or iterations reach
            "max_iterations").

        Args:
            noise_to_expval: Function mapping noise scale to expectation vales.
            max_iterations: Maximum number of iterations (optional).
                            Default: 100.
        Raises:
            ConvergenceWarning: iteration loop stops before convergence.
        """
        # Clear out the factory to make sure it is fresh.
        self.reset()

        counter = 0
        while not self.is_converged() and counter < max_iterations:
            next_param = self.next()
            next_result = noise_to_expval(next_param)
            self.push(next_param, next_result)
            counter += 1

        if counter == max_iterations:
            warnings.warn(
                "Factory iteration loop stopped before convergence. "
                f"Maximum number of iterations ({max_iterations}) "
                "was reached.",
                ConvergenceWarning,
            )

        return self

    def run(self, qp: QPROGRAM,
            executor: Callable[[QPROGRAM], float],
            scale_noise: Callable[[QPROGRAM, float], QPROGRAM],
            max_iterations: int = 100) -> 'Factory':
        """
        Runs the factory until convergence executing quantum circuits.
        Accepts different noise levels.

        Args:
            qp: Circuit to mitigate.
            executor: Function executing a circuit; returns an expectation
                      value.
            scale_noise: Function that scales the noise level of a quantum
                         circuit.
            max_iterations: Maximum number of iterations (optional).
                            Default: 100.
        """

        def _noise_to_expval(noise_param: float) -> float:
            """Evaluates the quantum expectation value for a given noise_
            param"""
            scaled_qp = scale_noise(qp, noise_param)
            return executor(scaled_qp)

        return self.iterate(_noise_to_expval, max_iterations)


class BatchedFactory(Factory):
    """
    Abstract class of a non-adaptive Factory.

    This is initialized with a given batch of "scale_factors".
    The "self.next" method trivially iterates over the elements of
    "scale_factors" in a non-adaptive way.
    Convergence is achieved when all the correpsonding expectation values have
    been measured.

    Specific (non-adaptive) zero-noise extrapolation algorithms can be derived
    from this class by overriding the "self.reduce" and (if necessary)
    the "__init__" method.

    Args:
        scale_factors: Iterable of noise scale factors at which
                       expectation values should be measured.
    Raises:
        ValueError: If the number of scale factors is less than 2.
        IndexError: If an iteration step fails.
    """

    def __init__(self, scale_factors: Iterable[float]) -> None:
        """Instantiates a new object of this Factory class."""
        if len(scale_factors) < 2:
            raise ValueError(
                "At least 2 scale factors are necessary."
            )
        self.scale_factors = scale_factors
        super(BatchedFactory, self).__init__()

    def next(self) -> float:
        try:
            next_param = self.scale_factors[len(self.outstack)]
        except IndexError:
            raise IndexError(
                "BatchedFactory cannot take another step. "
                "Number of batched scale_factors"
                f" ({len(self.scale_factors)}) exceeded."
            )
        return next_param

    def is_converged(self) -> bool:
        if len(self.outstack) != len(self.instack):
            raise IndexError(
                f"The length of 'self.instack' ({len(self.instack)}) "
                f"and 'self.outstack' ({len(self.outstack)}) must be equal."
            )
        return len(self.outstack) == len(self.scale_factors)


class PolyFactory(BatchedFactory):
    """
    Factory object implementing a zero-noise extrapolation algorithm based on
    a polynomial fit.

    Args:
        scale_factors: Iterable of noise scale factors at which
                       expectation values should be measured.
        order: Extrapolation order (degree of the polynomial fit).
               It cannot exceed len(scale_factors) - 1.
    Raises:
        ValueError: If data is not consistent with the extrapolation model.
        ExtrapolationWarning: If the extrapolation fit is ill-conditioned.
    Note:
        RichardsonFactory and LinearFactory are special cases of PolyFactory.
    """

    def __init__(self, scale_factors: Iterable[float], order: int) -> None:
        """Instantiates a new object of this Factory class."""

        if order > len(scale_factors) - 1:
            raise ValueError(
                "The extrapolation order cannot exceed len(scale_factors) - 1."
            )
        self.order = order
        super(PolyFactory, self).__init__(scale_factors)

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

        Args:
            instack: The array of noise scale factors.
            outstack: The array of expectation values.
            order: Extrapolation order (degree of the polynomial fit).
                   It cannot exceed len(scale_factors) - 1.
        Raises:
            ValueError: If data is not consistent with the extrapolation model.
            ExtrapolationWarning: If the extrapolation fit is ill-conditioned.
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
        coefficients = _mitiq_polyfit(instack, outstack, deg=order)
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
    """Factory object implementing Richardson's extrapolation.

    Args:
        scale_factors: Iterable of noise scale factors at which
                       expectation values should be measured.
    Raises:
        ValueError: If data is not consistent with the extrapolation model.
        ExtrapolationWarning: If the extrapolation fit is ill-conditioned.
    """

    def reduce(self) -> float:
        """Returns the Richardson's extrapolation to the zero-noise limit."""
        # Richardson's extrapolation is a particular case of a polynomial fit
        # with order equal to the number of data points minus 1.
        order = len(self.instack) - 1
        return PolyFactory.static_reduce(
            self.instack, self.outstack, order=order
        )


class LinearFactory(BatchedFactory):
    """
    Factory object implementing zero-noise extrapolation based
    on a linear fit.

    Args:
        scale_factors: Iterable of noise scale factors at which
                       expectation values should be measured.
    Raises:
        ValueError: If data is not consistent with the extrapolation model.
        ExtrapolationWarning: If the extrapolation fit is ill-conditioned.
    Example:
        >>> NOISE_LEVELS = [1.0, 2.0, 3.0]
        >>> fac = LinearFactory(NOISE_LEVELS)
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

    If y(x->inf) is unknown, the ansatz y(x) is fitted with a non-linear
    optimization.

    If y(x->inf) is given and avoid_log=False, the exponential
    model is mapped into a linear model by logarithmic transformation.

    Args:
        scale_factors: Iterable of noise scale factors at which
                       expectation values should be measured.
        asymptote: Infinite-noise limit (optional argument).
        avoid_log: If set to True, the exponential model is not linearized
                   with a logarithm and a non-linear fit is applied even
                   if asymptote is not None. The default value is False.
    Raises:
        ValueError: If data is not consistent with the extrapolation model.
        ExtrapolationError: If the extrapolation fit fails.
        ExtrapolationWarning: If the extrapolation fit is ill-conditioned.
    """

    def __init__(
            self, scale_factors: Iterable[float],
            asymptote: Optional[float] = None,
            avoid_log: bool = False,
    ) -> None:
        """Instantiate an new object of this Factory class."""
        super(ExpFactory, self).__init__(scale_factors)
        if not (asymptote is None or isinstance(asymptote, float)):
            raise ValueError(
                "The argument 'asymptote' must be either a float or None"
            )
        self.asymptote = asymptote
        self.avoid_log = avoid_log

    def reduce(self) -> float:
        """Returns the zero-noise limit"""
        return PolyExpFactory.static_reduce(self.instack,
                                            self.outstack,
                                            self.asymptote,
                                            order=1,
                                            avoid_log=self.avoid_log)[0]


class PolyExpFactory(BatchedFactory):
    """
    Factory object implementing a zero-noise extrapolation algorithm assuming
    an (almost) exponential ansatz with a non linear exponent, i.e.:

    y(x) = a + sign * exp(z(x)), where z(x) is a polynomial of a given order.

    The parameter "sign" is a sign variable which can be either 1 or -1,
    corresponding to decreasing and increasing exponentials, respectively.
    The parameter "sign" is automatically deduced from the data.

    If y(x->inf) is unknown, the ansatz y(x) is fitted with a non-linear
    optimization.

    If y(x->inf) is given and avoid_log=False, the exponential
    model is mapped into a polynomial model by logarithmic transformation.

    Args:
        scale_factors: Iterable of noise scale factors at which
                       expectation values should be measured.
        order: Extrapolation order (degree of the polynomial z(x)).
               It cannot exceed len(scale_factors) - 1.
               If asymptote is None, order cannot exceed
               len(scale_factors) - 2.
        asymptote: Infinite-noise limit (optional argument).
        avoid_log: If set to True, the exponential model is not linearized
                   with a logarithm and a non-linear fit is applied even
                   if asymptote is not None. The default value is False.
    Raises:
        ValueError: If data is not consistent with the extrapolation model.
        ExtrapolationError: If the extrapolation fit fails.
        ExtrapolationWarning: If the extrapolation fit is ill-conditioned.
    """

    def __init__(self,
                 scale_factors: Iterable[float],
                 order: int,
                 asymptote: Optional[float] = None,
                 avoid_log: bool = False) -> None:
        """Instantiates a new object of this Factory class."""
        super(PolyExpFactory, self).__init__(scale_factors)
        if not (asymptote is None or isinstance(asymptote, float)):
            raise ValueError(
                "The argument 'asymptote' must be either a float or None"
            )
        self.order = order
        self.asymptote = asymptote
        self.avoid_log = avoid_log

    @staticmethod
    def static_reduce(instack: List[float],
                      outstack: List[float],
                      asymptote: Optional[float],
                      order: int,
                      avoid_log: bool = False,
                      eps: float = 1.0e-6) -> Tuple[float, List[float]]:
        """
        Determines the zero-noise limit, assuming an exponential ansatz:
        y(x) = a + sign * exp(z(x)), where z(x) is a polynomial.

        The parameter "sign" is a sign variable which can be either 1 or -1,
        corresponding to decreasing and increasing exponentials, respectively.
        The parameter "sign" is automatically deduced from the data.

        It is also assumed that z(x-->inf)=-inf, such that y(x-->inf)-->a.

        If asymptote is None, the ansatz y(x) is fitted with a non-linear
        optimization.

        If asymptote is given and avoid_log=False, a linear fit with respect to
        z(x) := log[sign * (y(x) - asymptote)] is performed.

        This static method is equivalent to the "self.reduce" method
        of PolyExpFactory, but can be called also by other factories which are
        related to PolyExpFactory, e.g., ExpFactory, AdaExpFactory.

        Args:
            instack: x data values.
            outstack: y data values.
            asymptote: y(x->inf).
            order: Extrapolation order (degree of the polynomial z(x)).
            avoid_log: If set to True, the exponential model is not linearized
                       with a logarithm and a non-linear fit is applied even
                       if asymptote is not None. The default value is False.
            eps: Epsilon to regularize log(sign (instack - asymptote)) when
                 the argument is to close to zero or negative.
        Returns:
            (znl, params): Where "znl" is the zero-noise-limit and "params"
                           are the optimal fitting parameters.
        Raises:
            ValueError: If data is not consistent with the extrapolation model.
            ExtrapolationError: If the extrapolation fit fails.
            ExtrapolationWarning: If the extrapolation fit is ill-conditioned.
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
            raise ValueError("Extrapolation order is too high. "
                             "The order cannot exceed the number"
                             f" of data points minus {1 + shift}.")

        # Deduce "sign" parameter of the exponential ansatz
        slope, _ = _mitiq_polyfit(instack, outstack, deg=1)
        sign = np.sign(-slope)

        def _ansatz_unknown(x: float, *coeffs: float):
            """Ansatz of generic order with unknown asymptote."""
            # Coefficients of the polynomial to be exponentiated
            z_coeffs = coeffs[2:][::-1]
            return coeffs[0] + coeffs[1] * np.exp(x * np.polyval(z_coeffs, x))

        def _ansatz_known(x: float, *coeffs: float):
            """Ansatz of generic order with known asymptote."""
            # Coefficients of the polynomial to be exponentiated
            z_coeffs = coeffs[1:][::-1]
            return asymptote + coeffs[0] * np.exp(x * np.polyval(z_coeffs, x))

        # CASE 1: asymptote is None.
        if asymptote is None:
            # First guess for the parameter (decay or growth from "sign" to 0)
            p_zero = [0.0, sign, -1.0] + [0.0 for _ in range(order - 1)]
            opt_params = _mitiq_curve_fit(_ansatz_unknown,
                                          instack,
                                          outstack,
                                          p_zero)
            # The zero noise limit is ansatz(0)= asympt + b
            zero_limit = opt_params[0] + opt_params[1]
            return (zero_limit, opt_params)

        # CASE 2: asymptote is given and "avoid_log" is True
        if avoid_log:
            # First guess for the parameter (decay or growth from "sign")
            p_zero = [sign, -1.0] + [0.0 for _ in range(order - 1)]
            opt_params = _mitiq_curve_fit(_ansatz_known,
                                          instack,
                                          outstack,
                                          p_zero)
            # The zero noise limit is ansatz(0)= asymptote + b
            zero_limit = asymptote + opt_params[0]
            return (zero_limit, [asymptote] + list(opt_params))

        # CASE 3: asymptote is given and "avoid_log" is False
        # Polynomial fit of z(x).
        shifted_y = [max(sign * (y - asymptote), eps) for y in outstack]
        zstack = np.log(shifted_y)
        # Get coefficients {z_j} of z(x)= z_0 + z_1*x + z_2*x**2...
        # Note: coefficients are ordered from high powers to powers of x
        # Weights "w" are used to compensate for error propagation
        # after the log transformation y --> z
        z_coefficients = _mitiq_polyfit(instack,
                                        zstack,
                                        deg=order,
                                        weights=np.sqrt(np.abs(shifted_y)))
        # The zero noise limit is ansatz(0)
        zero_limit = asymptote + sign * np.exp(z_coefficients[-1])
        # Parameters from low order to high order
        params = [asymptote] + list(z_coefficients[::-1])
        return (zero_limit, params)

    def reduce(self) -> float:
        """Returns the zero-noise limit."""
        return self.static_reduce(self.instack,
                                  self.outstack,
                                  self.asymptote,
                                  self.order,
                                  self.avoid_log)[0]


class AdaExpFactory(Factory):
    """Factory object implementing an adaptive zero-noise extrapolation
    algorithm assuming an exponential ansatz y(x) = a + b * exp(-c * x),
    with c > 0.

    The noise scale factors are are chosen adaptively at each step,
    depending on the history of collected results.

    If y(x->inf) is unknown, the ansatz y(x) is fitted with a non-linear
    optimization.

    If y(x->inf) is given and avoid_log=False, the exponential
    model is mapped into a linear model by logarithmic transformation.

    Args:
        steps: The number of optimization steps. At least 3 are necessary.
        scale_factor: The second noise scale factor
                      (the first is always 1.0).
                      Further scale factors are adaptively determined.
        asymptote: The infinite noise limit (if known) of the expectation
                   value. Default is None.
        avoid_log: If set to True, the exponential model is not linearized
                   with a logarithm and a non-linear fit is applied even
                   if asymptote is not None. The default value is False.
    Raises:
        ValueError: If data is not consistent with the extrapolation model.
        ExtrapolationError: If the extrapolation fit fails.
        ExtrapolationWarning: If the extrapolation fit is ill-conditioned.
    """

    _SHIFT_FACTOR = 1.27846

    def __init__(
            self,
            steps: int,
            scale_factor: float = 2,
            asymptote: Optional[float] = None,
            avoid_log: bool = False
    ) -> None:
        """Instantiate a new object of this Factory class."""
        super(AdaExpFactory, self).__init__()
        if not (asymptote is None or isinstance(asymptote, float)):
            raise ValueError(
                "The argument 'asymptote' must be either a float or None"
            )
        if scale_factor <= 1:
            raise ValueError(
                "The argument 'scale_factor' must be strictly larger than one."
            )
        if steps < 3 + int(asymptote is None):
            raise ValueError("The argument 'steps' must be an integer"
                             " greater or equal to 3. "
                             "If 'asymptote' is None, 'steps' must be"
                             " greater or equal to 4.")
        self.steps = steps
        self.scale_factor = scale_factor
        self.asymptote = asymptote
        self.avoid_log = avoid_log
        # Keep a log of the optimization process storing:
        # noise value(s), expectation value(s), parameters, and zero limit
        self.history = [
        ]  # type: List[Tuple[List[float], List[float], List[float], float]]

    def next(self) -> float:
        """Returns the next noise level to execute a circuit at."""
        # The 1st scale factor is always 1
        if len(self.instack) == 0:
            return 1.0
        # The 2nd scale factor is self.scale_factor
        if len(self.instack) == 1:
            return self.scale_factor
        # If asymptote is None we use 2 * scale_factor as third noise parameter
        if (len(self.instack) == 2) and (self.asymptote is None):
            return 2 * self.scale_factor

        with warnings.catch_warnings():
            # This is an intermediate fit, so we suppress its warning messages
            warnings.simplefilter("ignore", ExtrapolationWarning)
            # Call reduce() to fit the exponent and save it in self.history
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
        """Returns the zero-noise limit."""
        zero_limit, params = PolyExpFactory.static_reduce(
            self.instack,
            self.outstack,
            self.asymptote,
            order=1,
            avoid_log=self.avoid_log,
        )
        # Update optimization history
        self.history.append((self.instack, self.outstack, params, zero_limit))
        return zero_limit
