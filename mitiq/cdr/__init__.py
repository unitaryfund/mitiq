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

"""Clifford data regression method for error mitigation.

TODO: Add reference + more information.
"""

from mitiq.cdr.cdr import execute_with_cdr
from mitiq.cdr.data_regression import (
    linear_fit_function,
    linear_fit_function_no_intercept,
)
from mitiq.cdr.clifford_training_data import (
    count_non_cliffords,
    is_clifford,
    generate_training_circuits,
)
