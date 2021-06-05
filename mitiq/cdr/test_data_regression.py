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

"""Tests for the data regression portion of Clifford data regression."""
import pytest
import numpy as np

from mitiq.cdr.data_regression import linear_fit_function


@pytest.mark.parametrize("nvariables", [1, 3])
def test_linear_fit_function(nvariables):
    a = np.random.RandomState(1).rand(nvariables)
    b = np.ones(nvariables + 1)
    assert linear_fit_function(a, b) == sum(a) + 1
