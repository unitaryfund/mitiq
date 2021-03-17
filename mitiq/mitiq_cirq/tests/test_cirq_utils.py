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
""" Tests for Cirq executors defined in cirq_utils.py"""

import numpy as np
import cirq


from mitiq.mitiq_cirq.cirq_utils import (execute)

def test_execute():
    """Tests if the executor function for Cirq returns a proper
    expectation value when an observable is provided."""

    # Test 1 using an observable
    qc = cirq.Circuit()
    qc += [cirq.X(cirq.LineQubit(0)), cirq.CNOT(cirq.LineQubit(0), cirq.LineQubit(1))]
    observable1 = np.diag([1, 0, 0, 0])
    observable_exp_value = execute(qc, obs=observable1)
    assert 0.0 == observable_exp_value

    # Test 2 using another observable
    new_qc = cirq.Circuit()
    new_qc += [cirq.X(cirq.LineQubit(0)), cirq.CNOT(cirq.LineQubit(0), cirq.LineQubit(1))]
    observable2 = np.diag([0, 0, 0, 1])
    observable_exp_value = execute(qc, obs=observable2)
    assert 1.0 == observable_exp_value
