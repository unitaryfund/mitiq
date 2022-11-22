# Copyright (C) 2022 Unitary Fund
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

"""Unit tests for inverse confusion matrix helper functions."""

from math import isclose
import numpy as np

from mitiq._typing import MeasurementResult
from mitiq.rem.inverse_confusion_matrix import (
    bitstrings_to_probability_vector,
    mitigate_measurements,
    sample_probability_vector,
)


def test_sample_probability_vector_single_qubit():
    bitstrings = sample_probability_vector(np.array([1, 0]), 10)
    assert all([b == [0] for b in bitstrings])

    bitstrings = sample_probability_vector(np.array([0, 1]), 10)
    assert all([b == [1] for b in bitstrings])

    bitstrings = sample_probability_vector(np.array([0.5, 0.5]), 1000)
    assert isclose(sum([b[0] for b in bitstrings]), 500, rel_tol=0.1)


def test_sample_probability_vector_two_qubits():
    bitstrings = sample_probability_vector(np.array([1, 0, 0, 0]), 10)
    assert all([b == [0, 0] for b in bitstrings])

    bitstrings = sample_probability_vector(np.array([0, 1, 0, 0]), 10)
    assert all([b == [0, 1] for b in bitstrings])

    bitstrings = sample_probability_vector(np.array([0, 0, 1, 0]), 10)
    assert all([b == [1, 0] for b in bitstrings])

    bitstrings = sample_probability_vector(np.array([0, 0, 0, 1]), 10)
    assert all([b == [1, 1] for b in bitstrings])


def test_bitstrings_to_probability_vector():
    pv = bitstrings_to_probability_vector([[0]])
    assert (pv == np.array([1, 0])).all()

    pv = bitstrings_to_probability_vector([[1]])
    assert (pv == np.array([0, 1])).all()

    pv = bitstrings_to_probability_vector([[0], [1]])
    assert (pv == np.array([0.5, 0.5])).all()

    pv = bitstrings_to_probability_vector([[0, 0]])
    assert (pv == np.array([1, 0, 0, 0])).all()

    pv = bitstrings_to_probability_vector([[1, 1]])
    assert (pv == np.array([0, 0, 0, 1])).all()


def test_probability_vector_roundtrip():
    for _ in range(10):
        pv = np.random.rand(4)
        pv /= np.sum(pv)
        assert isclose(
            np.linalg.norm(
                pv
                - bitstrings_to_probability_vector(
                    sample_probability_vector(pv, 1000)
                )
            ),
            0,
            abs_tol=0.1,
        )


def test_mitigate_measurements():
    identity = np.identity(4)

    measurements = MeasurementResult([[1, 0]])
    assert mitigate_measurements(measurements, identity) == measurements
    assert mitigate_measurements(measurements, np.flipud(identity)).result == [
        [0, 1]
    ]

    measurements = MeasurementResult([[0, 1]])
    assert mitigate_measurements(measurements, identity) == measurements
    assert mitigate_measurements(measurements, np.flipud(identity)).result == [
        [1, 0]
    ]
