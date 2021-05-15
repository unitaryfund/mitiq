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

"""Unit tests for scaling noise by identity insertion."""

import numpy as np
import pytest
from cirq import (
    Circuit,
    LineQubit,
    ops,
    testing,
    GateOperation,
)

from cirq.ops import IdentityGate
from mitiq.utils import _equal
from mitiq.zne.scaling.identity_insertion import (
    UnscalableCircuitError,
    _create_scale_mask,
    scale_gates_from_left,
    scale_gates_from_right,
    scale_gates_at_random,
)


@pytest.mark.parametrize(
    "scale_method",
    [scale_gates_at_random, scale_gates_from_left, scale_gates_from_right,],
)
def test_scaling_with_bad_scale_factor(scale_method):
    """Checks the test fails when scale factor is negative and input method is
     called directly."""
    with pytest.raises(ValueError, match="Requires scale_factor >= 0.0"):
        scale_method(Circuit(), scale_factor=-1.0)


@pytest.mark.parametrize("method", ["at_random", "from_right", "from_left"])
def test_create_mask_with_bad_scale_factor(method):
    """Checks the test fails when scale factor is negative and input method is
    a string."""
    with pytest.raises(ValueError, match="Requires scale_factor >= 0.0"):
        _create_scale_mask([1], scale_factor=-1.0, scaling_method=method)


def test_create_mask_with_bad_scaling_method():
    "Checks test fails when no scaling method is given."
    with pytest.raises(ValueError, match="'scaling_method' is not valid."):
        _create_scale_mask([1], scale_factor=1.5, scaling_method=None)


@pytest.mark.parametrize(
    "scale_method",
    [scale_gates_from_left, scale_gates_from_right, scale_gates_at_random],
)
def test_scale_from_left_no_stretch(scale_method):
    """Unit test for scaling gates from left for a scale factor of 0 i.e
    no identity gates are inserted."""
    circuit = testing.random_circuit(qubits=2, n_moments=10, op_density=0.99)
    scaled = scale_method(circuit, scale_factor=0,seed=None)
    assert _equal(scaled, circuit)
    assert not (scaled is circuit)


@pytest.mark.parametrize(
    "scale_method",
    [scale_gates_from_left, scale_gates_from_right, scale_gates_at_random],
)
def test_scale_from_left_and_right_small_factor(scale_method):
    """Basic test for scaling from left and right with small
    scale factor.
    """
    # Test Circuit
    # 0: ───H───@───────
    #           |
    # 1: ───H───X───T───
    qreg = LineQubit.range(2)
    circ = Circuit(
        [
            ops.H.on_each(*qreg),
            ops.CNOT.on(qreg[0], qreg[1]),
            ops.T.on(qreg[1]),
        ]
    )

    scaled = scale_method(circ, scale_factor=0.7, seed=3)
    correct = Circuit(
        [
            ops.H.on_each(*qreg),
            ops.I.on_each(*qreg),
            ops.CNOT.on(qreg[0], qreg[1]),
            GateOperation(IdentityGate(2), qreg),
            ops.T.on(qreg[1]),
            ops.I.on(qreg[1]),
        ]
    )
    assert _equal(scaled, correct)


@pytest.mark.parametrize(
    "scale_method",
    [scale_gates_from_left, scale_gates_from_right, scale_gates_at_random],
)
def test_scale_from_left_and_right_intermediate_factor(scale_method):
    """Basic test for scaling from left and right with intermediate
    scale factor.
    """
    # Test Circuit
    # 0: ───H───@───────
    #           |
    # 1: ───H───X───T───
    qreg = LineQubit.range(2)
    circ = Circuit(
        [
            ops.H.on_each(*qreg),
            ops.CNOT.on(qreg[0], qreg[1]),
            ops.T.on(qreg[1]),
        ]
    )

    scaled = scale_method(circ, scale_factor=1.5, seed=3)
    correct = Circuit(
        [
            ops.H.on_each(*qreg),
            ops.I.on_each(*qreg),
            ops.CNOT.on(qreg[0], qreg[1]),
            GateOperation(IdentityGate(2), qreg),
            ops.T.on(qreg[1]),
            ops.I.on(qreg[1]),
        ]
    )
    assert _equal(scaled, correct)


@pytest.mark.parametrize(
    "scale_method",
    [scale_gates_from_left, scale_gates_from_right, scale_gates_at_random],
)
def test_scale_from_left_and_right_full_factor(scale_method):
    """Basic test for scaling from left and right with full
    scale factor.
    """
    # Test Circuit
    # 0: ───H───@───────
    #           |
    # 1: ───H───X───T───
    qreg = LineQubit.range(2)
    circ = Circuit(
        [
            ops.H.on_each(*qreg),
            ops.CNOT.on(qreg[0], qreg[1]),
            ops.T.on(qreg[1]),
        ]
    )

    scaled = scale_method(circ, scale_factor=2.0, seed=3)
    correct = Circuit(
        [
            ops.H.on_each(*qreg),
            [ops.I.on_each(*qreg)] * 2,
            ops.CNOT.on(qreg[0], qreg[1]),
            [GateOperation(IdentityGate(2), qreg)] * 2,
            ops.T.on(qreg[1]),
            [ops.I.on(qreg[1])] * 2,
        ]
    )
    assert _equal(scaled, correct)


@pytest.mark.parametrize(
    "scale_method",
    [scale_gates_from_left, scale_gates_from_right, scale_gates_at_random],
)
def test_scale_from_left_and_right_three_qubits(scale_method):
    """Unit test for scaling gates from left to for a 3 qubit circuit."""
    # Test Circuit
    # 0: ───H───@───@───
    #           │   │
    # 1: ───H───X───@───
    #               │
    # 2: ───H───T───X───

    qreg = LineQubit.range(3)
    circ = Circuit(
        [
            ops.H.on_each(*qreg),
            ops.CNOT.on(qreg[0], qreg[1]),
            ops.T.on(qreg[2]),
            ops.TOFFOLI.on(*qreg),
        ]
    )

    scaled = scale_method(circ, scale_factor=2,seed=3 )
    correct = Circuit(
        [
            ops.H.on_each(*qreg),
            [ops.I.on_each(*qreg)] * 2,
            ops.CNOT.on(qreg[0], qreg[1]),
            [GateOperation(IdentityGate(2), (qreg[0], qreg[1]))] * 2,
            ops.T.on(qreg[2]),
            [ops.I.on(qreg[2])] * 2,
            ops.TOFFOLI.on(*qreg),
            [GateOperation(IdentityGate(3), qreg)] * 2,
        ]
    )
    assert _equal(scaled, correct)


@pytest.mark.parametrize(
    "scale_method",
    [scale_gates_from_left, scale_gates_from_right, scale_gates_at_random],
)
def test_scale_from_left_and_right_scale_factor_more_than_full_factor(
    scale_method,
):
    """Tests scaling from left with a scale_factor larger than two."""
    qreg = LineQubit.range(2)
    circuit = Circuit([ops.SWAP.on(*qreg)], [ops.CNOT.on(*qreg)])
    scaled = scale_method(circuit, scale_factor=5.0)
    correct = Circuit(
        [ops.SWAP.on(*qreg)],
        [GateOperation(IdentityGate(2), qreg)] * 5,
        [ops.CNOT.on(*qreg)],
        [GateOperation(IdentityGate(2), qreg)] * 5,
    )
    assert _equal(scaled, correct)


@pytest.mark.parametrize(
    "scale_method",
    [scale_gates_from_left, scale_gates_from_right, scale_gates_at_random],
)
def test_scale_from_left_and_right_with_terminal_measurements_min_stretch(
    scale_method,
):
    """Tests scaling from left with terminal measurements."""
    qreg = LineQubit.range(3)
    circ = Circuit(
        [ops.H.on_each(*qreg)],
        [ops.CNOT.on(qreg[0], qreg[1])],
        [ops.T.on(qreg[2])],
        [ops.TOFFOLI.on(*qreg)],
        [ops.measure_each(*qreg)],
    )
    scaled = scale_method(circ, scale_factor=0.0)
    correct = Circuit(
        [ops.H.on_each(*qreg)],
        [ops.CNOT.on(qreg[0], qreg[1])],
        [ops.T.on(qreg[2])],
        [ops.TOFFOLI.on(*qreg)],
        [ops.measure_each(*qreg)],
    )
    assert _equal(scaled, correct)


@pytest.mark.parametrize(
    "scale_method",
    [scale_gates_from_left, scale_gates_from_right, scale_gates_at_random],
)
def test_fold_from_left_and_right_with_terminal_measurements_max_stretch(
    scale_method,
):
    """Tests scaling from left with terminal measurements."""
    qreg = LineQubit.range(3)
    circ = Circuit(
        [ops.H.on_each(*qreg)],
        [ops.CNOT.on(qreg[0], qreg[1])],
        [ops.T.on(qreg[2])],
        [ops.TOFFOLI.on(*qreg)],
        [ops.measure_each(*qreg)],
    )
    scaled = scale_method(circ, scale_factor=2.0)
    correct = Circuit(
        [ops.H.on_each(*qreg)],
        [ops.I.on_each(*qreg)] * 2,
        [ops.CNOT.on(qreg[0], qreg[1])],
        [GateOperation(IdentityGate(2), (qreg[0], qreg[1]))] * 2,
        [ops.T.on(qreg[2])],
        [ops.I.on(qreg[2])] * 2,
        [ops.TOFFOLI.on(*qreg)],
        [GateOperation(IdentityGate(3), (qreg))] * 2,
        [ops.measure_each(*qreg)],
    )
    assert _equal(scaled, correct)

    # Make sure original circuit is not modified
    original = Circuit(
        [ops.H.on_each(*qreg)],
        [ops.CNOT.on(qreg[0], qreg[1])],
        [ops.T.on(qreg[2])],
        [ops.TOFFOLI.on(*qreg)],
        [ops.measure_each(*qreg)],
    )
    assert _equal(circ, original)


@pytest.mark.parametrize(
    "scale_method",
    [scale_gates_from_left, scale_gates_from_right, scale_gates_at_random,],
)
def test_scale_with_intermediate_measurements_raises_error(scale_method):
    """Tests local scaling functions raise an error on circuits with
    intermediate measurements.
    """
    qbit = LineQubit(0)
    circ = Circuit([ops.H.on(qbit)], [ops.measure(qbit)], [ops.T.on(qbit)])
    with pytest.raises(
        UnscalableCircuitError,
        match="Circuit contains intermediate measurements",
    ):
        scale_method(circ, scale_factor=3.0)


@pytest.mark.parametrize(
    "scale_method",
    [scale_gates_from_left, scale_gates_from_right, scale_gates_at_random],
)
def test_scale_with_channels_raises_error(scale_method):
    """Tests local scaling functions raise an error on circuits with
    non-unitary channels (which are not measurements).
    """
    qbit = LineQubit(0)
    circ = Circuit(
        ops.H.on(qbit), ops.depolarize(p=0.1).on(qbit), ops.measure(qbit)
    )
    with pytest.raises(
        UnscalableCircuitError, match="Circuit contains non-unitary channels"
    ):
        scale_method(circ, scale_factor=3.0)


@pytest.mark.parametrize(
    "scale_method",
    [scale_gates_from_left, scale_gates_from_right, scale_gates_at_random]
)
def test_scale_no_repeats(scale_method):
    """Tests scaling at random to ensure that no gates are folded twice and
    scaled gates are not scaled again.
    """
    qreg = LineQubit.range(2)
    circ = Circuit(
        [ops.H.on_each(qreg[0])],
        [ops.CNOT.on(*qreg)],
        [ops.X.on(qreg[1])],
        [ops.Y.on(qreg[0])],
        [ops.CZ.on(*qreg)],
    )
    circuit_ops = set(circ.all_operations())

    for scale in np.linspace(1.0, 3.0, 5):
        scaled = scale_method(circ, scale_factor=scale, seed=1)
        gates = list(scaled.all_operations())
        counts = {gate: gates.count(gate) for gate in circuit_ops}
        assert all(count <= 3 for count in counts.values())


def test_scale_right_retains_terminal_measurements_in_input_circuit():
    """Tests that scaling from the right doesn't modify the terminal
    measurements in the input circuit.
    """
    qbit = LineQubit(1)
    circ = Circuit(ops.H.on(qbit), ops.measure(qbit))
    scaled = scale_gates_from_right(circ, scale_factor=0.0)
    assert _equal(circ, scaled)


def test_local_scaling_methods_match_on_even_scale_factors():
    circuit = testing.random_circuit(
        qubits=3, n_moments=5, op_density=1.0, random_state=11
    )
    for s in (2, 6, 14):
        assert _equal(
            scale_gates_from_left(circuit, s),
            scale_gates_from_right(circuit, s),
            require_qubit_equality=True,
            require_measurement_equality=True,
        )

    for s in (2, 6, 14):
        assert _equal(
            scale_gates_from_left(circuit, s),
            scale_gates_at_random(circuit, s),
            require_qubit_equality=True,
            require_measurement_equality=True,
        )
