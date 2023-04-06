# Copyright (C) 2023 Unitary Fund
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


"""Tests for QPE benchmarking circuits."""

import pytest
import numpy as np
import cirq
from mitiq.utils import _equal
from mitiq.benchmarks.qpe_circuits import generate_qpe_circuit
from mitiq import SUPPORTED_PROGRAM_TYPES


def test_bad_qubit_number():
    for n in (-1, 0):
        with pytest.raises(
            ValueError,
            match="{} is invalid for the number of eigenvalue register\
             qubits.",
        ):
            generate_qpe_circuit(n, cirq.T)


def test_bad_gate():
    for g in (cirq.CNOT, cirq.SWAP):
        with pytest.raises(
            ValueError,
            match="This QPE circuit generation method only works for \
            1-qubit gates.",
        ):
            generate_qpe_circuit(4, g)


def test_bad_evalue_register():
    for g in (cirq.X, cirq.T):
        with pytest.raises(
            ValueError,
            match="The eigenvalue register must be larger than the eigenstate\
             register.",
        ):
            generate_qpe_circuit(1, g)


@pytest.mark.parametrize("return_type", SUPPORTED_PROGRAM_TYPES.keys())
def test_conversion(return_type):
    circuit = generate_qpe_circuit(3, cirq.Y, return_type)
    assert return_type in circuit.__module__
