# Copyright (C) 2021 Unitary Fund
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

"""The data regression portion of Clifford data regression."""
from typing import Sequence

import numpy as np
import numpy.typing as npt


def linear_fit_function(
    x_data: npt.NDArray[np.float64], params: Sequence[float]
) -> float:
    """Returns y(x) = a_1 x_1 + ... + a_n x_n + b.

    Args:
        x_data: The independent variables x_1, ..., x_n. In CDR, these are
            nominally the noisy expectation values to perform regression on.
        params: Parameters a_1, ..., a_n, b of the linear fit. Note the b
            parameter is the intercept of the fit.
    """
    return sum(a * x for a, x in zip(params, x_data)) + params[-1]


def linear_fit_function_no_intercept(
    x_data: npt.NDArray[np.float64], params: Sequence[float]
) -> float:
    """Returns y(x) = a_1 x_1 + ... + a_n x_n.

    Args:
        x_data: The independent variables x_1, ..., x_n. In CDR, these are
            nominally the noisy expectation values to perform regression on.
        params: Parameters a_1, ..., a_n of the linear fit.
    """
    return sum(a * x for a, x in zip(params, x_data))
