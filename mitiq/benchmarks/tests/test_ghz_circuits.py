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

"""Tests for GHZ circuits."""

import pytest
import numpy as np

from mitiq.benchmarks import ghz_circuits
from mitiq._typing import SUPPORTED_PROGRAM_TYPES


@pytest.mark.parametrize("nqubits", [1, 5])
def test_ghz_circuits(nqubits):
    # test GHZ creation
    circuit = ghz_circuits.generate_ghz_circuit(nqubits)
    # check that the state |0...0> and the state |1...1>
    # both have probability close to 0.5
    sv = circuit.final_state_vector()
    zero_prob = abs(sv[0]) ** 2
    one_prob = abs(sv[-1]) ** 2
    assert np.isclose(zero_prob, 0.5)
    assert np.isclose(one_prob, 0.5)


def test_ghz_value_error():
    # test GHZ error raising
    with pytest.raises(ValueError):
        circuit = ghz_circuits.generate_ghz_circuit(0)  # noqa: F841


@pytest.mark.parametrize("return_type", SUPPORTED_PROGRAM_TYPES.keys())
def test_ghz_conversion(return_type):
    circuit = ghz_circuits.generate_ghz_circuit(3, return_type)
    assert return_type in circuit.__module__
