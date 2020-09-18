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

"""Functions for fitting data."""
from typing import Callable, List, Optional, Sequence
import warnings

import numpy as np
from numpy.lib.polynomial import RankWarning
from scipy.optimize import curve_fit, OptimizeWarning


class ExtrapolationError(Exception):
    pass


_EXTR_ERR = (
    "The extrapolation fit failed to converge."
    " The problem may be solved by switching to a more stable"
    " extrapolation model such as `LinearFactory`."
)


class ExtrapolationWarning(Warning):
    pass


_EXTR_WARN = (
    "The extrapolation fit may be ill-conditioned."
    " Likely, more data points are necessary to fit the parameters"
    " of the model."
)


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



