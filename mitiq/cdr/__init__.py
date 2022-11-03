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

"""Clifford data regression method for error mitigation as introduced in:

[1] Piotr Czarnik, Andrew Arramsmith, Patrick Coles, Lukasz Cincio,
    "Error mitigation with Clifford quantum circuit data,"
    (https://arxiv.org/abs/2005.10189).
[2] Angus Lowe, Max Hunter Gordon, Piotr Czarnik, Andrew Arramsmith,
    Patrick Coles, Lukasz Cincio, "Unified approach to data-driven error
    mitigation," (https://arxiv.org/abs/2011.01157).
"""

from mitiq.cdr.data_regression import (
    linear_fit_function,
    linear_fit_function_no_intercept,
)
from mitiq.cdr.clifford_training_data import (
    generate_training_circuits,
)
from mitiq.cdr.cdr import execute_with_cdr, mitigate_executor, cdr_decorator
