# Copyright (C) 2020 Unitary Fund
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

"""Tests for randomized benchmarking circuits."""

import pytest
import numpy as np

from mitiq.benchmarks.randomized_benchmarking import generate_rb_circuits
from mitiq._typing import SUPPORTED_PROGRAM_TYPES


@pytest.mark.parametrize("n_qubits", (1, 2))
def test_rb_circuits(n_qubits):
    depth = 10

    # test single qubit RB
    for trials in [2, 3]:
        circuits = generate_rb_circuits(
            n_qubits=n_qubits, num_cliffords=depth, trials=trials
        )
        for qc in circuits:
            # we check the ground state population to ignore any global phase
            wvf = qc.final_state_vector()
            zero_prob = abs(wvf[0] ** 2)
            assert np.isclose(zero_prob, 1)


@pytest.mark.parametrize("n_qubits", (1, 2))
@pytest.mark.parametrize("return_type", SUPPORTED_PROGRAM_TYPES.keys())
def test_rb_conversion(n_qubits, return_type):
    depth = 10
    for trials in [2, 3]:
        circuits = generate_rb_circuits(
            n_qubits=n_qubits,
            num_cliffords=depth,
            trials=trials,
            return_type=return_type,
        )
        for qc in circuits:
            assert return_type in qc.__module__
