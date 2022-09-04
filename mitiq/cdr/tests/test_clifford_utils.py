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

"""Tests for generating (near) Clifford circuits."""
import pytest
import numpy as np

import cirq
from cirq.circuits import Circuit

from mitiq._typing import SUPPORTED_PROGRAM_TYPES
from mitiq.interface import convert_from_mitiq
from mitiq.cdr.clifford_utils import (
    is_clifford_angle,
    is_clifford,
    closest_clifford,
    random_clifford,
    angle_to_proximity,
    angle_to_proximities,
    probabilistic_angle_to_clifford,
    count_non_cliffords,
    _CLIFFORD_ANGLES,
)


@pytest.mark.parametrize("circuit_type", SUPPORTED_PROGRAM_TYPES.keys())
def test_is_clifford_with_clifford(circuit_type):
    circuit = convert_from_mitiq(
        cirq.Circuit(cirq.Z.on(cirq.LineQubit(0))), circuit_type
    )
    assert is_clifford(circuit)


@pytest.mark.parametrize("circuit_type", SUPPORTED_PROGRAM_TYPES.keys())
def test_is_clifford_with_nonclifford(circuit_type):
    circuit = convert_from_mitiq(
        cirq.Circuit(cirq.T.on(cirq.LineQubit(0))), circuit_type
    )
    assert not is_clifford(circuit)


@pytest.mark.parametrize("circuit_type", SUPPORTED_PROGRAM_TYPES.keys())
def test_count_non_cliffords(circuit_type):
    a, b = cirq.LineQubit.range(2)
    circuit = Circuit(
        cirq.rz(0.0).on(a),  # Clifford.
        cirq.rx(0.1 * np.pi).on(b),  # Non-Clifford.
        cirq.rx(0.5 * np.pi).on(b),  # Clifford
        cirq.rz(0.4 * np.pi).on(b),  # Non-Clifford.
        cirq.rz(0.5 * np.pi).on(b),  # Clifford.
        cirq.CNOT.on(a, b),  # Clifford.
    )
    circuit = convert_from_mitiq(circuit, circuit_type)

    assert count_non_cliffords(circuit) == 2


def test_count_non_cliffords_empty_circuit():
    assert count_non_cliffords(Circuit()) == 0


def test_is_clifford_angle():
    for p in range(4):
        assert is_clifford_angle(p * np.array(_CLIFFORD_ANGLES)).all()

    assert not is_clifford_angle(-0.17)


def test_closest_clifford():
    for ang in _CLIFFORD_ANGLES:
        angs = np.linspace(ang - np.pi / 4 + 0.01, ang + np.pi / 4 - 0.01)
        for a in angs:
            assert closest_clifford(a) == ang


def test_random_clifford():
    assert set(random_clifford(20, np.random.RandomState(1))).issubset(
        _CLIFFORD_ANGLES
    )


def test_angle_to_proximities():
    for sigma in np.linspace(0.1, 2, 10):
        for ang in _CLIFFORD_ANGLES:
            probabilities = angle_to_proximities(ang, sigma)
            assert (isinstance(p, float) for p in probabilities)


def test_angle_to_proximity():
    for sigma in np.linspace(0.1, 2, 10):
        probabilities = angle_to_proximity(_CLIFFORD_ANGLES, sigma)
        assert all(isinstance(p, float) for p in probabilities)


def test_probabilistic_angles_to_clifford():
    for sigma in np.linspace(0.1, 2, 10):
        angles = probabilistic_angle_to_clifford(
            _CLIFFORD_ANGLES, sigma, np.random.RandomState(1)
        )
        assert all(a in _CLIFFORD_ANGLES for a in angles)
