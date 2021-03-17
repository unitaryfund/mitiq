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
from cirq import LineQubit

from mitiq.mitiq_cirq.cirq_utils import (execute)

def test_execute():
    """Tests if the simulation executor function for Cirq returns a property
    expectation value when an observable is provided."""

    qc = Circuit()
    qc += [cirq.X(LineQubit(0)), cirq.CNOT(LineQubit(0), LineQubit(1))]
    observable_exp_value = execute(qc, obs=np.diag([1, 0, 0, 0]))
    assert 0.0 == observable_exp_value

    new_qc = = Circuit()
    new_qc += [cirq.X(LineQubit(0)), cirq.CNOT(LineQubit(0), LineQubit(1))]
    observable_exp_value = execute(qc, obs=np.diag([0, 0, 0, 1]))
    assert 1.0 == observable_exp_value
