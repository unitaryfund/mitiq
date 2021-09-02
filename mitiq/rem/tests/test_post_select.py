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

from mitiq.rem import post_select, MeasurementResult


def test_post_select():
    res = MeasurementResult(
        [[0, 1, 1], [1, 1, 0], [0, 0, 0], [0, 0, 1], [1, 1, 1]]
    )

    assert post_select(res, lambda bits: sum(bits) == 3).result == [[1, 1, 1]]
    assert post_select(res, lambda bits: sum(bits) == 2).result == [
        [0, 1, 1],
        [1, 1, 0],
    ]
    assert post_select(res, lambda bits: sum(bits) == 1).result == [[0, 0, 1]]
    assert post_select(res, lambda bits: sum(bits) == 0).result == [[0, 0, 0]]


def test_post_select_inverted():
    res = MeasurementResult([[0, 0, 1], [1, 1, 0], [0, 0, 0]])

    assert post_select(res, lambda bits: sum(bits) == 1).result == [[0, 0, 1]]
    assert post_select(
        res, lambda bits: sum(bits) == 1, inverted=True
    ).result == [[1, 1, 0], [0, 0, 0],]

    assert post_select(res, lambda bits: sum(bits) == 3).result == []
    assert (
        post_select(res, lambda bits: sum(bits) == 3, inverted=True).result
        == res.result
    )


@pytest.mark.parametrize("inverted", (True, False))
def test_post_select_empty_measurement_result(inverted):
    res = MeasurementResult([])
    assert post_select(res, lambda bits: sum(bits) == 3, inverted).result == []


def test_post_select_hamming_weight_less_than():
    res = MeasurementResult(
        [[0, 1, 1], [1, 1, 0], [0, 0, 0], [0, 0, 1], [1, 1, 1]]
    )

    assert post_select(res, lambda bits: sum(bits) < 1).result == [[0, 0, 0]]
    assert post_select(res, lambda b: sum(b) < 2).result == [
        [0, 0, 0],
        [0, 0, 1],
    ]


def test_post_select_hamming_weight_specific_bit():
    res = MeasurementResult(
        [[0, 1, 1], [1, 1, 0], [0, 0, 0], [0, 0, 1], [1, 1, 1]]
    )

    assert post_select(res, lambda b: b[1] == 0).result == [
        [0, 0, 0],
        [0, 0, 1],
    ]


def test_post_select_edge_cases():
    samples = MeasurementResult([[1], [0], [1], [0], [0], [1], [1], [1]])

    assert post_select(samples, lambda bits: sum(bits) == -1).result == []
    assert post_select(samples, lambda bits: sum(bits) == 23).result == []
