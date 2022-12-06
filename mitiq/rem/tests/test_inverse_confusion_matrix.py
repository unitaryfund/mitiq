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

from functools import reduce
from math import isclose
import numpy as np
import pytest

from mitiq._typing import MeasurementResult
from mitiq.rem.inverse_confusion_matrix import (
    bitstrings_to_probability_vector,
    generate_inverse_confusion_matrix,
    generate_tensored_inverse_confusion_matrix,
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


def test_generate_inverse_confusion_matrix():
    num_qubits = 2
    identity = np.identity(4)
    assert (
        generate_inverse_confusion_matrix(num_qubits, p0=0, p1=0) == identity
    ).all()
    assert (
        generate_inverse_confusion_matrix(num_qubits, p0=1, p1=1)
        == np.flipud(identity)
    ).all()


@pytest.mark.parametrize(
    "num_qubits, confusion_matrices, expected",
    [
        (2, [np.identity(2), np.identity(2)], np.identity(4)),
        (2, [np.identity(4)], np.identity(4)),
        (3, [np.identity(4), np.identity(2)], np.identity(8)),
        # all are faulty qubits, flipping to opposite value
        (
            2,
            [np.flipud(np.identity(2)), np.flipud(np.identity(2))],
            np.flipud(np.identity(4)),
        ),
        (
            3,
            [np.flipud(np.identity(4)), np.flipud(np.identity(2))],
            np.flipud(np.identity(8)),
        ),
        # wrongly sized confusion matrices
        (3, [np.identity(2), np.identity(2)], ValueError),
        # one qubit flips values, one is perfect, one is random
        (
            3,
            [np.flipud(np.identity(2)), np.identity(2), 0.5 * np.ones((2, 2))],
            np.linalg.pinv(
                reduce(
                    np.kron,
                    [
                        np.flipud(np.identity(2)),
                        np.identity(2),
                        0.5 * np.ones((2, 2)),
                    ],
                )
            ),
        ),
    ],
)
def test_generate_tensored_inverse_confusion_matrix(
    num_qubits, confusion_matrices, expected
):
    if expected is ValueError:
        with pytest.raises(ValueError):
            generate_tensored_inverse_confusion_matrix(
                num_qubits, confusion_matrices
            )
    else:
        assert np.isclose(
            generate_tensored_inverse_confusion_matrix(
                num_qubits, confusion_matrices
            ),
            expected,
        ).all()


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
