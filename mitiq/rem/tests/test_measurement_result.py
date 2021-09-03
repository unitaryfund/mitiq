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

"""Unit tests for measurement results."""
import pytest

import numpy as np
from mitiq.rem.measurement_result import MeasurementResult


@pytest.mark.parametrize("asarray", (True, False))
@pytest.mark.parametrize("qubit_indices", ((0, 1), (1, 20)))
def test_measurement_result(asarray, qubit_indices):
    bitstrings = [[0, 0], [0, 1], [1, 0]]
    if asarray:
        bitstrings = np.array(bitstrings)
    result = MeasurementResult(bitstrings, qubit_indices=qubit_indices)

    assert result.nqubits == 2
    assert result.qubit_indices == qubit_indices
    assert result.shots == 3
    assert np.allclose(result.result, bitstrings)
    assert (
        repr(result) == f"MeasurementResult(result=[[0, 0], [0, 1], [1, 0]], "
        f"qubit_indices={qubit_indices})"
    )


def test_measurement_result_bad_qubit_indices():
    with pytest.raises(ValueError, match="MeasurementResult has"):
        MeasurementResult([[0], [1]], qubit_indices=(1, 5))


def test_measurement_result_not_bits():
    with pytest.raises(ValueError, match="contains elements"):
        MeasurementResult(result=[[2]])

    with pytest.raises(ValueError, match="contains elements"):
        MeasurementResult(result=[[0, 0], [0, 1], [-1, 0]])


def test_getitem():
    result = MeasurementResult([[0, 0, 1], [0, 1, 0], [1, 0, 0]])
    assert np.allclose(result[[0]], np.array([[0], [0], [1]]))
    assert np.allclose(result[[1]], np.array([[0], [1], [0]]))
    assert np.allclose(result[[2]], np.array([[1], [0], [0]]))

    assert np.allclose(result[[0, 1]], np.array([[0, 0], [0, 1], [1, 0]]))
    assert np.allclose(result[[0, 2]], np.array([[0, 1], [0, 0], [1, 0]]))
    assert np.allclose(result[[1, 2]], np.array([[0, 1], [1, 0], [0, 0]]))


def test_empty():
    result = MeasurementResult([])
    assert result.nqubits == 0
    assert result.shots == 0
    assert result.result == []


def test_iter():
    bitstrings = [[0, 1, 1], [0, 1, 0], [1, 0, 0]]

    for m, bits in zip(MeasurementResult(bitstrings), bitstrings):
        assert m == bits


def test_convert_to_array():
    bitstrings = [[0, 0], [0, 1], [1, 0]]
    result = MeasurementResult(bitstrings)
    assert np.allclose(result.asarray, np.array(bitstrings))

    assert np.allclose(MeasurementResult([]).asarray, np.array([]))
