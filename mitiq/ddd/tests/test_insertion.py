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

"""Unit tests for DDD slack windows and DDD insertion tools."""

import numpy as np
import cirq
from mitiq.ddd.insertion import get_circuit_mask

test_mask_one = np.array(
    [
        [1, 0, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0, 0, 0],
        [0, 0, 1, 1, 0, 0, 0],
        [0, 0, 0, 1, 1, 0, 0],
        [0, 0, 0, 0, 1, 1, 0],
        [0, 0, 0, 0, 0, 1, 1],
        [0, 0, 0, 0, 0, 0, 1],
    ]
)

test_mask_two = np.array(
    [
        [1, 0],
        [1, 1],
        [1, 0],
        [1, 0],
        [1, 0],
        [1, 0],
        [1, 0],
        [1, 0],
        [1, 0],
        [1, 0],
    ]
)


def test_get_circuit_mask_one():
    circuit = cirq.Circuit(
        cirq.SWAP(q, q + 1) for q in cirq.LineQubit.range(7)
    )
    circuit_mask = get_circuit_mask(circuit)
    assert np.allclose(circuit_mask, test_mask_one)


def test_get_circuit_mask_two():
    qreg = cirq.GridQubit.rect(10, 1)
    circuit = cirq.Circuit(cirq.ops.H.on_each(*qreg), cirq.ops.H.on(qreg[1]))
    circuit_mask = get_circuit_mask(circuit)
    assert np.allclose(circuit_mask, test_mask_two)
