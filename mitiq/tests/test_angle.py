"""Unit tests for Param Cirq circuits."""

from copy import deepcopy

import numpy as np
import pytest
from cirq import Circuit, GridQubit, LineQubit, ops, inverse
from cirq import rx, ry, CNOT, X, Y, Z

from mitiq.utils import _equal
from mitiq.angle import (
    add_parameter_noise
)

def test_identity_scale_1q():
    # tests that when scale factor = 1, the circuit is the 
    # same. 
    qreg = LineQubit.range(3)
    circ = Circuit(
        [ops.X.on_each(qreg)],
        [ops.Y.on(qreg[0])]
    )
    scaled = add_parameter_noise(circ, scale_factor=1, sigma=0.001)
    assert _equal(circ, scaled)

def test_non_identity_scale_1q():
    # tests that when scale factor = 1, the circuit is the 
    # same. 
    qreg = LineQubit.range(3)
    circ = Circuit(
        [ops.rx(np.pi*1.0).on_each(qreg)],
        [ops.ry(np.pi*1.0).on(qreg[0])]
    )
    np.random.seed(42)
    stretch = 2
    base_noise = 0.001
    noises = np.random.normal(loc=0.0, scale=np.sqrt((stretch-1)*base_noise), size = (4,))
    np.random.seed(42)

    scaled = add_parameter_noise(circ, scale_factor=stretch, sigma=base_noise)
    result = []
    for moment in scaled:
        for op in moment.operations:
            gate = deepcopy(op.gate)
            param = gate.exponent
            result.append(param*np.pi - np.pi)
    assert np.all(np.isclose(result - noises, 0))

def test_identity_scale_2q():
    # tests that when scale factor = 1, the circuit is the 
    # same. 
    qreg = LineQubit.range(2)
    circ = Circuit(
        [ops.CNOT.on(qreg[0], qreg[1])]
    )
    scaled = add_parameter_noise(circ, scale_factor=1, sigma=0.001)
    assert _equal(circ, scaled)

def test_non_identity_scale_2q():
    # tests that when scale factor = 1, the circuit is the 
    # same. 
    qreg = LineQubit.range(2)
    circ = Circuit(
        [ops.CNOT.on(qreg[0], qreg[1])]
    )
    np.random.seed(42)
    stretch = 2
    base_noise = 0.001
    noises = np.random.normal(loc=0.0, scale=np.sqrt((stretch-1)*base_noise), size = (1,))
    np.random.seed(42)
    scaled = add_parameter_noise(circ, scale_factor=stretch, sigma=base_noise)
    result = []
    for moment in scaled:
        for op in moment.operations:
            gate = deepcopy(op.gate)
            param = gate.exponent
            result.append(param*np.pi - np.pi)
    assert np.all(np.isclose(result - noises, 0))


def test_scale_with_measurement():
    """We should ignore measurement gates."""
    
    # Test circuit:
    # 0: ───H───T───@───M───
    #               │   │
    # 1: ───H───M───┼───┼───
    #               │   │
    # 2: ───H───────X───M───
    qreg = LineQubit.range(3)
    circ = Circuit(
        [ops.H.on_each(qreg)],
        [ops.T.on(qreg[0])],
        [ops.measure(qreg[1])],
        [ops.CNOT.on(qreg[0], qreg[2])],
        [ops.measure(qreg[0], qreg[2])],
    )
    scaled = add_parameter_noise(circ, scale_factor=1, sigma=0.001)
    assert _equal(circ, scaled)
