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

"""Tests for volume circuits. The cirq functions that do the main work 
are tested here:
cirq-core/cirq/contrib/quantum_volume/quantum_volume_test.py

Tests below check that generate_volume_circuit() works for mitiq's interface.
"""

import pytest
import numpy as np

import cirq

from mitiq.benchmarks.volume_circuits import generate_volume_circuit 
from mitiq._typing import SUPPORTED_PROGRAM_TYPES

def test_generate_model_circuit():
    """Test that random circuit of the right length is generated."""
    circuit, _ = generate_volume_circuit(3, 3)
    assert len(circuit) == 3


@pytest.mark.parametrize("return_type", SUPPORTED_PROGRAM_TYPES.keys())
def test_volume_conversion(return_type):
    circuit, _ = generate_volume_circuit(3, 3, return_type)
    assert return_type in circuit.__module__


