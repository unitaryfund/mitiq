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

"""Tests for W-state benchmarking circuits."""

import pytest
import numpy as np
import cirq
from mitiq.utils import _equal

from mitiq.benchmarks.w_state_circuits import GRotationGate, W_circuit_linear_complexity

def test_w4_circuit():
    output_circuit = W_circuit_linear_complexity(4)
    qubits = cirq.LineQubit.range(4)
    correct_circuit = cirq.Circuit(
        cirq.ControlledGate(GRotationGate(1/4)).on(qubits[0], qubits[1]), cirq.CNOT(qubits[1], qubits[0]), 
        cirq.ControlledGate(GRotationGate(1/3)).on(qubits[1], qubits[2]), cirq.CNOT(qubits[2], qubits[1]),
        cirq.ControlledGate(GRotationGate(1/2)).on(qubits[2], qubits[3]), cirq.CNOT(qubits[3], qubits[2]),
        )
    assert _equal(output_circuit, correct_circuit, require_qubit_equality=True)
