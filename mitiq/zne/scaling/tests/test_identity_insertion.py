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

"""Unit tests for scaling noise by inserting identity layers."""
import numpy as np
import pytest
from mitiq.zne.scaling.identity_insertion import (
    UnscalableCircuitError,
    _calculate_id_layers,
    insert_id_layers,
)
from cirq import Circuit, ops, LineQubit
from mitiq.utils import _equal


@pytest.mark.parametrize("scale_factor", (1, 2, 3, 4, 5, 6))
def test_id_layers_whole_scale_factor(scale_factor):
    """Tests if n-1 identity layers are inserted uniformly when
    the intended scale factor is n."""
    qreg = LineQubit.range(3)
    circ = Circuit(
        [ops.H.on_each(*qreg)],
        [ops.CNOT.on(qreg[0], qreg[1])],
        [ops.X.on(qreg[2])],
        [ops.TOFFOLI.on(*qreg)],
    )
    scaled = insert_id_layers(circ, scale_factor=scale_factor)
    num_layers = scale_factor - 1
    correct = Circuit(
        [ops.H.on_each(*qreg)],
        [ops.I.on_each(*qreg)] * num_layers,
        [ops.CNOT.on(qreg[0], qreg[1])],
        [ops.X.on(qreg[2])],
        [ops.I.on_each(*qreg)] * num_layers,
        [ops.TOFFOLI.on(*qreg)],
        [ops.I.on_each(*qreg)] * num_layers,
    )
    assert _equal(scaled, correct)
