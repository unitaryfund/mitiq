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

"""Unit tests for post-selection of measurement results."""
import pytest

from mitiq.rem import post_select


def test_post_select():
    measurement = [[0, 1, 1], [1, 1, 0], [0, 0, 0], [0, 0, 1], [1, 1, 1]]

    assert post_select(measurement, hamming_weight=3) == [[1, 1, 1]]
    assert post_select(measurement, hamming_weight=2) == [[0, 1, 1], [1, 1, 0]]
    assert post_select(measurement, hamming_weight=1) == [[0, 0, 1]]
    assert post_select(measurement, hamming_weight=0) == [[0, 0, 0]]


def test_post_select_inverted():
    res = [[0, 0, 1], [1, 1, 0], [0, 0, 0]]

    assert post_select(res, hamming_weight=1) == [[0, 0, 1]]
    assert post_select(res, hamming_weight=1, inverted=True) == [[1, 1, 0]]

    assert post_select(res, hamming_weight=3) == []
    assert post_select(res, hamming_weight=3, inverted=True) == [[0, 0, 0]]


@pytest.mark.parametrize("inverted", (True, False))
def test_post_select_empty_measurement_result(inverted):
    assert post_select([], hamming_weight=3, inverted=inverted) == []


def test_post_select_edge_cases():
    samples = [[1], [0], [1], [0], [0], [1], [1], [1]]

    assert post_select(samples, hamming_weight=-1) == []
    assert post_select(samples, hamming_weight=23) == []
