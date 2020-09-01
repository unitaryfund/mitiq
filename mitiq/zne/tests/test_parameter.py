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

"""Unit tests for Param Cirq circuits."""

from copy import deepcopy

import pytest
import numpy as np
from cirq import Circuit, LineQubit, ops, CSWAP

from mitiq.utils import _equal
from mitiq.zne.parameter import (
    scale_parameters,
    _get_base_gate,
    GateTypeException
)


def test_identity_scale_1q():
    """Tests that when scale factor = 1, the circuit is the
    same.
    """
    qreg = LineQubit.range(3)
    circ = Circuit(
        [ops.X.on_each(qreg)],
        [ops.Y.on(qreg[0])]
    )
    scaled = scale_parameters(circ, scale_factor=1, sigma=0.001)
    assert _equal(circ, scaled)


def test_non_identity_scale_1q():
    """Tests that when scale factor = 1, the circuit is the
    same.
    """
    qreg = LineQubit.range(3)
    circ = Circuit(
        [ops.rx(np.pi*1.0).on_each(qreg)],
        [ops.ry(np.pi*1.0).on(qreg[0])]
    )
    np.random.seed(42)
    stretch = 2
    base_noise = 0.001
    noises = np.random.normal(loc=0.0, scale=np.sqrt(
        (stretch-1)*base_noise), size=(4,))
    np.random.seed(42)

    scaled = scale_parameters(
        circ, scale_factor=stretch, sigma=base_noise, seed=42)
    result = []
    for moment in scaled:
        for op in moment.operations:
            gate = deepcopy(op.gate)
            param = gate.exponent
            result.append(param*np.pi - np.pi)
    assert np.all(np.isclose(result - noises, 0))


def test_identity_scale_2q():
    """Tests that when scale factor = 1, the circuit is the
    same.
    """
    qreg = LineQubit.range(2)
    circ = Circuit(
        [ops.CNOT.on(qreg[0], qreg[1])]
    )
    scaled = scale_parameters(circ, scale_factor=1, sigma=0.001)
    assert _equal(circ, scaled)


def test_non_identity_scale_2q():
    """Tests that when scale factor = 1, the circuit is the
    same.
    """
    qreg = LineQubit.range(2)
    circ = Circuit(
        [ops.CNOT.on(qreg[0], qreg[1])]
    )
    np.random.seed(42)
    stretch = 2
    base_noise = 0.001
    noises = np.random.normal(loc=0.0, scale=np.sqrt(
        (stretch-1)*base_noise), size=(1,))
    np.random.seed(42)
    scaled = scale_parameters(
        circ, scale_factor=stretch, sigma=base_noise, seed=42)
    result = []
    for moment in scaled:
        for op in moment.operations:
            gate = deepcopy(op.gate)
            param = gate.exponent
            result.append(param*np.pi - np.pi)
    assert np.all(np.isclose(result - noises, 0))


def test_scale_with_measurement():
    """Tests that we ignore measurement gates.

    Test circuit:
    0: ───H───T───@───M───
                  │   │
    1: ───H───M───┼───┼───
                  │   │
    2: ───H───────X───M───

    """
    qreg = LineQubit.range(3)
    circ = Circuit(
        [ops.H.on_each(qreg)],
        [ops.T.on(qreg[0])],
        [ops.measure(qreg[1])],
        [ops.CNOT.on(qreg[0], qreg[2])],
        [ops.measure(qreg[0], qreg[2])],
    )
    scaled = scale_parameters(circ, scale_factor=1, sigma=0.001)
    assert _equal(circ, scaled)


def test_gate_type():
    qreg = LineQubit.range(3)
    allowed_op = ops.H.on(qreg[0])
    _get_base_gate(allowed_op.gate)
    with pytest.raises(GateTypeException):
        forbidden_op = CSWAP(qreg[0], qreg[1], qreg[2])
        _get_base_gate(forbidden_op)
