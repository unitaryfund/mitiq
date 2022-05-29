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

"""Tests for quantum volume circuits. The Cirq functions that do the main work
are tested here:
cirq-core/cirq/contrib/quantum_volume/quantum_volume_test.py

Tests below check that generate_volume_circuit() works as a wrapper and
fits with mitiq's interface.
"""

import pytest

import cirq

from mitiq.benchmarks.volume_circuits import (
    generate_volume_circuit,
    compute_heavy_bitstrings,
)

from mitiq._typing import SUPPORTED_PROGRAM_TYPES


def test_generate_model_circuit_no_seed():
    """Test that random circuit of the right length is generated."""
    circuit, _ = generate_volume_circuit(3, 3)
    assert len(circuit) == 3


def test_generate_model_circuit_with_seed():
    """Test that a model circuit is determined by its seed."""
    circuit_1, _ = generate_volume_circuit(3, 3, seed=1)
    circuit_2, _ = generate_volume_circuit(3, 3, seed=1)
    circuit_3, _ = generate_volume_circuit(3, 3, seed=2)

    assert circuit_1 == circuit_2
    assert circuit_2 != circuit_3


def test_compute_heavy_bitstrings():
    """Test that the heavy bitstrings can be computed from a given circuit."""
    a, b, c = cirq.LineQubit.range(3)
    model_circuit = cirq.Circuit(
        [
            cirq.Moment([]),
            cirq.Moment([cirq.X(a), cirq.Y(b)]),
            cirq.Moment([]),
            cirq.Moment([cirq.CNOT(a, c)]),
            cirq.Moment([cirq.Z(a), cirq.H(b)]),
        ]
    )
    true_heavy_set = [[1, 0, 1], [1, 1, 1]]
    computed_heavy_set = compute_heavy_bitstrings(model_circuit, 3)
    assert computed_heavy_set == true_heavy_set


@pytest.mark.parametrize("return_type", SUPPORTED_PROGRAM_TYPES.keys())
def test_volume_conversion(return_type):
    circuit, _ = generate_volume_circuit(3, 3, return_type=return_type)
    assert return_type in circuit.__module__
