# Copyright (C) 2020 Unitary Fund
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Classes corresponding to different zero-noise extrapolation methods."""
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Callable, Dict, List, Optional, Sequence, Tuple
import warnings

import numpy as np
from numpy.lib.polynomial import RankWarning
from scipy.optimize import curve_fit, OptimizeWarning

from mitiq import QPROGRAM
from mitiq.utils import _are_close_dict


def _instack_to_scale_factors(instack: List[Dict[str, float]]) -> List[float]:
    """Extracts a list of scale factors from a list of dictionaries."""
    if not all(isinstance(params, dict) for params in instack):
        raise ValueError("instack must be a list of dictionaries")
    return [params["scale_factor"] for params in instack]


class ExtrapolationError(Exception):
    """Error raised by :class:`.Factory` objects when
    the extrapolation fit fails.
    """

    pass


_EXTR_ERR = (
    "The extrapolation fit failed to converge."
    " The problem may be solved by switching to a more stable"
    " extrapolation model such as `LinearFactory`."
)


class ExtrapolationWarning(Warning):
    """Warning raised by :class:`.Factory` objects when
    the extrapolation fit is ill-conditioned.
    """

    pass


_EXTR_WARN = (
    "The extrapolation fit may be ill-conditioned."
    " Likely, more data points are necessary to fit the parameters"
    " of the model."
)


class ConvergenceWarning(Warning):
    """Warning raised by :class:`.Factory` objects when
    their `iterate` method fails to converge.
    """

    pass


def mitiq_curve_fit(
    ansatz: Callable[..., float],
    scale_factors: Sequence[float],
    exp_values: Sequence[float],
    init_params: Optional[List[float]] = None,
) -> List[float]:
    """This is a wrapping of the `scipy.optimize.curve_fit` function with
    custom errors and warnings. It is used to make a non-linear fit.

    Args:
        ansatz : The model function used for zero-noise extrapolation.
                 The first argument is the noise scale variable,
                 the remaining arguments are the parameters to fit.
        scale_factors: The array of noise scale factors.
        exp_values: The array of expectation values.
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
            opt_params, _ = curve_fit(
                ansatz, scale_factors, exp_values, p0=init_params
            )
        for warn in warn_list:
            # replace OptimizeWarning with ExtrapolationWarning
            if warn.category is OptimizeWarning:
                warn.category = ExtrapolationWarning
                warn.message = _EXTR_WARN  # type: ignore
            # re-raise all warnings
            warnings.warn_explicit(
                warn.message, warn.category, warn.filename, warn.lineno
            )
    except RuntimeError:
        raise ExtrapolationError(_EXTR_ERR) from None
    return list(opt_params)


def mitiq_polyfit(
    scale_factors: Sequence[float],
    exp_values: Sequence[float],
    deg: int,
    weights: Optional[Sequence[float]] = None,
) -> List[float]:
    """This is a wrapping of the `numpy.polyfit` function with
    custom warnings. It is used to make a polynomial fit.

    Args:
        scale_factors: The array of noise scale factors.
        exp_values: The array of expectation values.
        deg: The degree of the polynomial fit.
        weights: Optional array of weights for each sampled point.
                 This is used to make a weighted least squares fit.

    Returns:
        opt_params: The array of optimal parameters.

    Raises:
        ExtrapolationWarning: If the extrapolation fit is ill-conditioned.
    """
    with warnings.catch_warnings(record=True) as warn_list:
        opt_params = np.polyfit(scale_factors, exp_values, deg, w=weights)
    for warn in warn_list:
        # replace RankWarning with ExtrapolationWarning
        if warn.category is RankWarning:
            warn.category = ExtrapolationWarning
            warn.message = _EXTR_WARN  # type: ignore
        # re-raise all warnings
        warnings.warn_explicit(
            warn.message, warn.category, warn.filename, warn.lineno
        )
    return list(opt_params)


class Factory(ABC):
    """Abstract class designed to adaptively produce a new noise scaling
    parameter based on a historical stack of previous noise scale parameters
    ("self._instack") and previously estimated expectation values
    ("self._outstack").

    Specific zero-noise extrapolation algorithms, adaptive or non-adaptive,
    are derived from this class.

    A Factory object is not supposed to directly perform any quantum
    computation, only the classical results of quantum experiments are
    processed by it.
    """

    def __init__(self) -> None:
        """Initialization arguments (e.g. noise scale factors) depend on the
        particular extrapolation algorithm and can be added to the "__init__"
        method of the associated derived class.
        """
        self._instack: List[Dict[str, float]] = []
        self._outstack: List[float] = []
        self.opt_params: List[float] = []

    def push(self, instack_val: dict, outstack_val: float) -> None:
        """Appends "instack_val" to "self._instack" and "outstack_val" to
        "self._outstack". Each time a new expectation value is computed this
        method should be used to update the internal state of the Factory.
        """
        self._instack.append(instack_val)
        self._outstack.append(outstack_val)

    def get_scale_factors(self) -> np.ndarray:
        """Returns the scale factors at which the factory has computed
        expectation values.
        """
        return np.array(
            [params.get("scale_factor") for params in self._instack]
        )

    def get_expectation_values(self) -> np.ndarray:
        """Returns the expectation values computed by the factory."""
        return np.array(self._outstack)

    @abstractmethod
    def next(self) -> Dict[str, float]:
        """Returns a dictionary of parameters to execute a circuit at."""
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
        """Resets the instack, outstack, and optimal parameters of the Factory
        to empty lists.
        """
        self._instack = []
        self._outstack = []
        self.opt_params = []

    def iterate(
        self, noise_to_expval: Callable[..., float], max_iterations: int = 100,
    ) -> "Factory":
        """Evaluates a sequence of expectation values until enough
        data is collected (or iterations reach "max_iterations").

        Args:
            noise_to_expval: Function mapping a noise scale factor to an
                             expectation value. If shot_list is not None,
                             "shot" must be an argument of the function.
            max_iterations: Maximum number of iterations (optional).
                            Default: 100.

        Raises:
            ConvergenceWarning: If iteration loop stops before convergence.
        """
        # Reset the instack, outstack, and optimal parameters
        self.reset()

        counter = 0
        while not self.is_converged() and counter < max_iterations:
            next_in_params = self.next()
            next_exec_params = deepcopy(next_in_params)

            # Get next scale factor and remove it from next_exec_params
            scale_factor = next_exec_params.pop("scale_factor")
            next_expval = noise_to_expval(scale_factor, **next_exec_params)
            self.push(next_in_params, next_expval)
            counter += 1

        if counter == max_iterations:
            warnings.warn(
                "Factory iteration loop stopped before convergence. "
                f"Maximum number of iterations ({max_iterations}) "
                "was reached.",
                ConvergenceWarning,
            )

        return self

    def run(
        self,
        qp: QPROGRAM,
        executor: Callable[..., float],
        scale_noise: Callable[[QPROGRAM, float], QPROGRAM],
        num_to_average: int = 1,
        max_iterations: int = 100,
    ) -> "Factory":
        """Evaluates a sequence of expectation values by executing quantum
        circuits until enough data is collected (or iterations reach
        "max_iterations").

        Args:
            qp: Circuit to mitigate.
            executor: Function executing a circuit; returns an expectation
                value. If shot_list is not None, then "shot" must be
                an additional argument of the executor.
            scale_noise: Function that scales the noise level of a quantum
                circuit.
            num_to_average: Number of times expectation values are computed by
                the executor after each call to scale_noise, then averaged.
            max_iterations: Maximum number of iterations (optional).
        """

        def _noise_to_expval(scale_factor, **exec_params) -> float:
            """Evaluates the quantum expectation value for a given
            scale_factor and other executor parameters."""
            expectation_values = []
            for _ in range(num_to_average):
                scaled_qp = scale_noise(qp, scale_factor)
                expectation_values.append(executor(scaled_qp, **exec_params))
            return np.average(expectation_values)

        return self.iterate(_noise_to_expval, max_iterations)

    def __eq__(self, other):
        if len(self._instack) != len(other._instack):
            return False
        for dict_a, dict_b in zip(self._instack, other._instack):
            if not _are_close_dict(dict_a, dict_b):
                return False
        return np.allclose(self._outstack, other._outstack)


class BatchedFactory(Factory):
    """Abstract class of a non-adaptive Factory.

    This is initialized with a given batch of "scale_factors".
    The "self.next" method trivially iterates over the elements of
    "scale_factors" in a non-adaptive way.
    Convergence is achieved when all the correpsonding expectation values have
    been measured.

    Specific (non-adaptive) zero-noise extrapolation algorithms can be derived
    from this class by overriding the "self.reduce" and (if necessary)
    the "__init__" method.

    Args:
        scale_factors: Sequence of noise scale factors at which
                       expectation values should be measured.
        shot_list: Optional sequence of integers corresponding to the number
                   of samples taken for each expectation value. If this
                   argument is explicitly passed to the factory, it must have
                   the same length of scale_factors and the executor function
                   must accept "shots" as a valid keyword argument.

    Raises:
        ValueError: If the number of scale factors is less than 2.
        IndexError: If an iteration step fails.
    """

    def __init__(
        self,
        scale_factors: Sequence[float],
        shot_list: Optional[List[int]] = None,
    ) -> None:
        """Instantiates a new object of this Factory class."""
        if len(scale_factors) < 2:
            raise ValueError("At least 2 scale factors are necessary.")

        if shot_list and (
            not isinstance(shot_list, Sequence)
            or not all([isinstance(shots, int) for shots in shot_list])
        ):
            raise TypeError(
                "The optional argument shot_list must be None "
                "or a valid iterator of integers."
            )
        if shot_list and (len(scale_factors) != len(shot_list)):
            raise IndexError(
                "The arguments scale_factors and shot_list"
                " must have the same length."
                f" But len(scale_factors) is {len(scale_factors)}"
                f" and len(shot_list) is {len(shot_list)}."
            )

        self._scale_factors = scale_factors
        self._shot_list = shot_list

        super(BatchedFactory, self).__init__()

    def next(self) -> Dict[str, float]:
        """Returns a dictionary of parameters to execute a circuit at."""
        in_params = {}
        try:
            index = len(self._outstack)
            in_params["scale_factor"] = self._scale_factors[index]
            if self._shot_list:
                in_params["shots"] = self._shot_list[index]
        except IndexError:
            raise IndexError(
                "BatchedFactory cannot take another step. "
                "Number of batched scale_factors"
                f" ({len(self._scale_factors)}) exceeded."
            )
        return in_params

    def is_converged(self) -> bool:
        """Returns True if all needed expectation values have been computed,
        else False.
        """
        if len(self._outstack) != len(self._instack):
            raise IndexError(
                f"The length of 'self._instack' ({len(self._instack)}) "
                f"and 'self._outstack' ({len(self._outstack)}) must be equal."
            )
        return len(self._outstack) == len(self._scale_factors)

    def __eq__(self, other):
        return Factory.__eq__(self, other) and np.allclose(
            self._scale_factors, other._scale_factors
        )


class PolyFactory(BatchedFactory):
    """Factory object implementing a zero-noise extrapolation algorithm based on
    a polynomial fit.

    Args:
        scale_factors: Sequence of noise scale factors at which
                       expectation values should be measured.
        order: Extrapolation order (degree of the polynomial fit).
               It cannot exceed len(scale_factors) - 1.
        shot_list: Optional sequence of integers corresponding to the number
                   of samples taken for each expectation value. If this
                   argument is explicitly passed to the factory, it must have
                   the same length of scale_factors and the executor function
                   must accept "shots" as a valid keyword argument.

    Raises:
        ValueError: If data is not consistent with the extrapolation model.
        ExtrapolationWarning: If the extrapolation fit is ill-conditioned.

    Note:
        RichardsonFactory and LinearFactory are special cases of PolyFactory.
    """

    def __init__(
        self,
        scale_factors: Sequence[float],
        order: int,
        shot_list: Optional[List[int]] = None,
    ) -> None:
        """Instantiates a new object of this Factory class."""
        if order > len(scale_factors) - 1:
            raise ValueError(
                "The extrapolation order cannot exceed len(scale_factors) - 1."
            )
        self.order = order
        super(PolyFactory, self).__init__(scale_factors, shot_list)

    def reduce(self) -> float:
        """Returns the zero-noise limit found by fitting a polynomial of degree
        `self.order` to the input data of scale factors and expectation values.

        Stores the optimal parameters for the fit in `self.opt_params`.
        """
        scale_factors = self.get_scale_factors()
        expectation_values = self.get_expectation_values()

        if self.order > len(scale_factors) - 1:
            raise ValueError(
                f"Extrapolation order is too high. The order cannot exceed "
                f"len(self.get_scale_factors()) but order = {self.order} and "
                f"len(self.get_scale_factors()) = {len(scale_factors)}."
            )

        self.opt_params = mitiq_polyfit(
            scale_factors, expectation_values, self.order
        )
        return self.opt_params[-1]

    def __eq__(self, other):
        return BatchedFactory.__eq__(self, other) and self.order == other.order


class RichardsonFactory(BatchedFactory):
    """Factory object implementing Richardson's extrapolation.

    Args:
        scale_factors: Sequence of noise scale factors at which
                       expectation values should be measured.
        shot_list: Optional sequence of integers corresponding to the number
                   of samples taken for each expectation value. If this
                   argument is explicitly passed to the factory, it must have
                   the same length of scale_factors and the executor function
                   must accept "shots" as a valid keyword argument.

    Raises:
        ValueError: If data is not consistent with the extrapolation model.
        ExtrapolationWarning: If the extrapolation fit is ill-conditioned.
    """

    def reduce(self) -> float:
        """Returns the zero-noise limit found by Richardson's extrapolation.

        Stores the optimal parameters for the fit in `self.opt_params`.
        """
        # Richardson's extrapolation is a particular case of a polynomial fit
        # with order equal to the number of data points minus 1.
        self.opt_params = mitiq_polyfit(
            self.get_scale_factors(),
            self.get_expectation_values(),
            deg=len(self.get_scale_factors()) - 1,
        )
        return self.opt_params[-1]


class LinearFactory(BatchedFactory):
    """
    Factory object implementing zero-noise extrapolation based
    on a linear fit.

    Args:
        scale_factors: Sequence of noise scale factors at which
                       expectation values should be measured.
        shot_list: Optional sequence of integers corresponding to the number
                   of samples taken for each expectation value. If this
                   argument is explicitly passed to the factory, it must have
                   the same length of scale_factors and the executor function
                   must accept "shots" as a valid keyword argument.
    Raises:
        ValueError: If data is not consistent with the extrapolation model.
        ExtrapolationWarning: If the extrapolation fit is ill-conditioned.
    Example:
        >>> NOISE_LEVELS = [1.0, 2.0, 3.0]
        >>> fac = LinearFactory(NOISE_LEVELS)
    """

    def reduce(self) -> float:
        """Returns the zero-noise limit found by fitting a line to the input
        data of scale factors and expectation values.

        Stores the optimal parameters for the fit in `self.opt_params`.
        """
        # Linear extrapolation is a particular case of a polynomial fit
        # with order equal to 1.
        self.opt_params = mitiq_polyfit(
            self.get_scale_factors(), self.get_expectation_values(), deg=1
        )
        return self.opt_params[-1]


class ExpFactory(BatchedFactory):
    """
    Factory object implementing a zero-noise extrapolation algorithm assuming
    an exponential ansatz y(x) = a + b * exp(-c * x), with c > 0.

    If y(x->inf) is unknown, the ansatz y(x) is fitted with a non-linear
    optimization.

    If y(x->inf) is given and avoid_log=False, the exponential
    model is mapped into a linear model by logarithmic transformation.

    Args:
        scale_factors: Sequence of noise scale factors at which
                       expectation values should be measured.
        asymptote: Infinite-noise limit (optional argument).
        avoid_log: If set to True, the exponential model is not linearized
                   with a logarithm and a non-linear fit is applied even
                   if asymptote is not None. The default value is False.
        shot_list: Optional sequence of integers corresponding to the number
                   of samples taken for each expectation value. If this
                   argument is explicitly passed to the factory, it must have
                   the same length of scale_factors and the executor function
                   must accept "shots" as a valid keyword argument.

    Raises:
        ValueError: If data is not consistent with the extrapolation model.
        ExtrapolationError: If the extrapolation fit fails.
        ExtrapolationWarning: If the extrapolation fit is ill-conditioned.
    """

    def __init__(
        self,
        scale_factors: Sequence[float],
        asymptote: Optional[float] = None,
        avoid_log: bool = False,
        shot_list: Optional[List[int]] = None,
    ) -> None:
        """Instantiate an new object of this Factory class."""
        super(ExpFactory, self).__init__(scale_factors, shot_list)
        if not (asymptote is None or isinstance(asymptote, float)):
            raise ValueError(
                "The argument 'asymptote' must be either a float or None"
            )
        self.asymptote = asymptote
        self.avoid_log = avoid_log

    def reduce(self) -> float:
        """Returns the zero-noise limit found by fitting the exponential ansatz.

        Stores the optimal parameters for the fit in `self.opt_params`.
        """
        zne_value, self.opt_params = PolyExpFactory.static_reduce(
            self._instack,
            self._outstack,
            self.asymptote,
            order=1,
            avoid_log=self.avoid_log,
        )
        return zne_value

    def __eq__(self, other):
        if (
            self.asymptote
            and other.asymptote is None
            or self.asymptote is None
            and other.asymptote
        ):
            return False
        if self.asymptote is None and other.asymptote is None:
            return (
                BatchedFactory.__eq__(self, other)
                and self.avoid_log == other.avoid_log
            )
        return (
            BatchedFactory.__eq__(self, other)
            and np.isclose(self.asymptote, other.asymptote)
            and self.avoid_log == other.avoid_log
        )


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
        scale_factors: Sequence of noise scale factors at which
                       expectation values should be measured.
        order: Extrapolation order (degree of the polynomial z(x)).
               It cannot exceed len(scale_factors) - 1.
               If asymptote is None, order cannot exceed
               len(scale_factors) - 2.
        asymptote: Infinite-noise limit (optional argument).
        avoid_log: If set to True, the exponential model is not linearized
                   with a logarithm and a non-linear fit is applied even
                   if asymptote is not None. The default value is False.
        shot_list: Optional sequence of integers corresponding to the number
                   of samples taken for each expectation value. If this
                   argument is explicitly passed to the factory, it must have
                   the same length of scale_factors and the executor function
                   must accept "shots" as a valid keyword argument.

    Raises:
        ValueError: If data is not consistent with the extrapolation model.
        ExtrapolationError: If the extrapolation fit fails.
        ExtrapolationWarning: If the extrapolation fit is ill-conditioned.
    """

    def __init__(
        self,
        scale_factors: Sequence[float],
        order: int,
        asymptote: Optional[float] = None,
        avoid_log: bool = False,
        shot_list: Optional[List[int]] = None,
    ) -> None:
        """Instantiates a new object of this Factory class."""
        super(PolyExpFactory, self).__init__(scale_factors, shot_list)
        if not (asymptote is None or isinstance(asymptote, float)):
            raise ValueError(
                "The argument 'asymptote' must be either a float or None"
            )
        self.order = order
        self.asymptote = asymptote
        self.avoid_log = avoid_log

    def reduce(self) -> float:
        """Returns the zero-noise limit found by fitting the ansatz.

        Stores the optimal parameters for the fit in `self.opt_params`.
        """
        zne_value, self.opt_params = self.static_reduce(
            self._instack,
            self._outstack,
            self.asymptote,
            self.order,
            self.avoid_log,
        )
        return zne_value

    @staticmethod
    def static_reduce(
        instack: List[dict],
        exp_values: List[float],
        asymptote: Optional[float],
        order: int,
        avoid_log: bool = False,
        eps: float = 1.0e-6,
    ) -> Tuple[float, List[float]]:
        """Determines the zero-noise limit assuming an exponential ansatz.

        The exponential ansatz is y(x) = a + sign * exp(z(x)) where z(x) is a
        polynomial and "sign" is either +1 or -1 corresponding to decreasing
        and increasing exponentials, respectively. The parameter "sign" is
        automatically deduced from the data.

        It is also assumed that z(x-->inf) = -inf, such that y(x-->inf) --> a.

        If asymptote is None, the ansatz y(x) is fitted with a non-linear
        optimization.

        If asymptote is given and avoid_log=False, a linear fit with respect to
        z(x) := log[sign * (y(x) - asymptote)] is performed.

        This static method is equivalent to the "self.reduce" method
        of PolyExpFactory, but can be called also by other factories which are
        related to PolyExpFactory, e.g., ExpFactory, AdaExpFactory.

        Args:
            instack: The array of input dictionaries, where each
                     dictionary is supposed to have the key "scale_factor".
            exp_values: The array of expectation values.
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

        scale_factors = _instack_to_scale_factors(instack)

        # Check arguments
        error_str = (
            "Data is not enough: at least two data points are necessary."
        )
        if scale_factors is None or exp_values is None:
            raise ValueError(error_str)
        if len(scale_factors) != len(exp_values) or len(scale_factors) < 2:
            raise ValueError(error_str)
        if order > len(scale_factors) - (1 + shift):
            raise ValueError(
                "Extrapolation order is too high. "
                "The order cannot exceed the number"
                f" of data points minus {1 + shift}."
            )

        # Deduce "sign" parameter of the exponential ansatz
        slope, _ = mitiq_polyfit(scale_factors, exp_values, deg=1)
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
            opt_params = mitiq_curve_fit(
                _ansatz_unknown, scale_factors, exp_values, p_zero
            )
            # The zero noise limit is ansatz(0)= asympt + b
            zero_limit = opt_params[0] + opt_params[1]
            return (zero_limit, opt_params)

        # CASE 2: asymptote is given and "avoid_log" is True
        if avoid_log:
            # First guess for the parameter (decay or growth from "sign")
            p_zero = [sign, -1.0] + [0.0 for _ in range(order - 1)]
            opt_params = mitiq_curve_fit(
                _ansatz_known, scale_factors, exp_values, p_zero
            )
            # The zero noise limit is ansatz(0)= asymptote + b
            zero_limit = asymptote + opt_params[0]
            return (zero_limit, [asymptote] + list(opt_params))

        # CASE 3: asymptote is given and "avoid_log" is False
        # Polynomial fit of z(x).
        shifted_y = [max(sign * (y - asymptote), eps) for y in exp_values]
        zstack = np.log(shifted_y)

        # Get coefficients {z_j} of z(x)= z_0 + z_1*x + z_2*x**2...
        # Note: coefficients are ordered from high powers to powers of x
        # Weights "w" are used to compensate for error propagation
        # after the log transformation y --> z
        z_coefficients = mitiq_polyfit(
            scale_factors,
            zstack,
            deg=order,
            weights=np.sqrt(np.abs(shifted_y)),
        )
        # The zero noise limit is ansatz(0)
        zero_limit = asymptote + sign * np.exp(z_coefficients[-1])
        # Parameters from low order to high order
        params = [asymptote] + list(z_coefficients[::-1])
        return zero_limit, params

    def __eq__(self, other):
        return (
            BatchedFactory.__eq__(self, other)
            and np.isclose(self.asymptote, other.asymptote)
            and self.avoid_log == other.avoid_log
            and self.order == other.order
        )


# Keep a log of the optimization process storing:
# noise value(s), expectation value(s), parameters, and zero limit
OptimizationHistory = List[
    Tuple[List[Dict[str, float]], List[float], List[float], float]
]


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
        max_scale_factor: Maximum noise scale factor. Default is 6.0.
    Raises:
        ValueError: If data is not consistent with the extrapolation model.
        ExtrapolationError: If the extrapolation fit fails.
        ExtrapolationWarning: If the extrapolation fit is ill-conditioned.
    """

    _SHIFT_FACTOR = 1.27846

    def __init__(
        self,
        steps: int,
        scale_factor: float = 2.0,
        asymptote: Optional[float] = None,
        avoid_log: bool = False,
        max_scale_factor: float = 6.0,
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
            raise ValueError(
                "The argument 'steps' must be an integer"
                " greater or equal to 3. "
                "If 'asymptote' is None, 'steps' must be"
                " greater or equal to 4."
            )
        if max_scale_factor <= 1:
            raise ValueError(
                "The argument 'max_scale_factor' must be"
                " strictly larger than one."
            )
        self._steps = steps
        self._scale_factor = scale_factor
        self.asymptote = asymptote
        self.avoid_log = avoid_log
        self.max_scale_factor = max_scale_factor
        self.history: OptimizationHistory = []

    def next(self) -> Dict[str, float]:
        """Returns a dictionary of parameters to execute a circuit at."""
        # The 1st scale factor is always 1
        if len(self._instack) == 0:
            return {"scale_factor": 1.0}
        # The 2nd scale factor is self._scale_factor
        if len(self._instack) == 1:
            return {"scale_factor": self._scale_factor}
        # If asymptote is None we use 2 * scale_factor as third noise parameter
        if (len(self._instack) == 2) and (self.asymptote is None):
            return {"scale_factor": 2 * self._scale_factor}

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
        next_scale_factor = min(
            1.0 + self._SHIFT_FACTOR / np.abs(exponent), self.max_scale_factor
        )
        return {"scale_factor": next_scale_factor}

    def is_converged(self) -> bool:
        """Returns True if all the needed expectation values have been
        computed, else False.
        """
        if len(self._outstack) != len(self._instack):
            raise IndexError(
                f"The length of 'self._instack' ({len(self._instack)}) "
                f"and 'self._outstack' ({len(self._outstack)}) must be equal."
            )
        return len(self._outstack) == self._steps

    def reduce(self) -> float:
        """Returns the zero-noise limit."""
        zero_limit, self.opt_params = PolyExpFactory.static_reduce(
            self._instack,
            self._outstack,
            self.asymptote,
            order=1,
            avoid_log=self.avoid_log,
        )
        # Update optimization history
        self.history.append(
            (self._instack, self._outstack, self.opt_params, zero_limit)
        )
        return zero_limit

    def __eq__(self, other) -> bool:
        return (
            Factory.__eq__(self, other)
            and self._steps == other._steps
            and self._scale_factor == other._scale_factor
            and np.isclose(self.asymptote, other.asymptote)
            and self.avoid_log == other.avoid_log
            and np.allclose(self.history, other.history)
        )
