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

"""Unit tests for scaling by layer."""
import pytest
from cirq import (
    Circuit,
    LineQubit,
    ops,
    synchronize_terminal_measurements,
)
from mitiq.zne.scaling.layer_scaling import layer_folding, layer_folding_all


def test_layer_folding_with_measurements():
    # Test circuit
    # 0: ───H───M───────
    #
    # 1: ───H───@───M───
    #           │
    # 2: ───────X───M───
    q = LineQubit.range(3)
    circuit = Circuit(
        ops.H(q[0]),
        ops.H(q[1]),
        ops.CNOT(*q[1:]),
        ops.measure_each(*q),
    )
    folded_circuit = layer_folding(circuit, [1] * len(circuit))

    expected_folded_circuit = Circuit(
        [ops.H(q[1])] * 3,
        [ops.CNOT(*q[1:])] * 3,
        ops.measure_each(*q),
    )
    expected_folded_circuit = synchronize_terminal_measurements(
        expected_folded_circuit
    )
    assert folded_circuit == expected_folded_circuit


def test_layer_folding():
    # Test circuit
    # 0: ───H───@───@───
    #           │   │
    # 1: ───H───X───@───
    #               │
    # 2: ───H───X───X───
    qreg = LineQubit.range(3)
    circ = Circuit(
        [ops.H.on_each(*qreg)],
        [ops.CNOT.on(qreg[0], qreg[1])],
        [ops.X.on(qreg[2])],
        [ops.TOFFOLI.on(*qreg)],
    )

    # Iterate over every possible combination of layerwise folds for a maximum
    # number of 5-folds.
    total_folds = 5
    for i1 in range(total_folds):
        for i2 in range(total_folds):
            for i3 in range(total_folds):
                layers_to_fold = [i1, i2, i3]

                folded_circuit = layer_folding(circ, layers_to_fold)

                # For a given layer, the number of copies on a layer will be
                # 2n + 1 where "n" is the number of folds to perform.
                qreg = LineQubit.range(3)
                correct = Circuit(
                    # Layer-1
                    [ops.H.on_each(*qreg)] * (2 * (layers_to_fold[0]) + 1),
                    # Layer-2
                    [ops.CNOT.on(qreg[0], qreg[1])]
                    * (2 * (layers_to_fold[1]) + 1),
                    [ops.X.on(qreg[2])] * (2 * (layers_to_fold[1]) + 1),
                    # Layer-3
                    [ops.TOFFOLI.on(*qreg)] * (2 * (layers_to_fold[2]) + 1),
                )
                assert folded_circuit == correct


@pytest.mark.parametrize("num_folds", range(5))
def test_layer_folding_all(num_folds):
    # Test circuit
    # 0: ───H───@───
    #           │
    # 1: ───────X───
    q0, q1 = LineQubit.range(2)
    circuit = Circuit(
        [ops.H(q0)],
        [ops.CNOT(q0, q1)],
    )

    circuit_folded = layer_folding_all(circuit=circuit, num_folds=num_folds)

    # First element of list should consist of circuit with only first layer
    # folded.
    expected_circuit_folded_1 = Circuit(
        [ops.H(q0)] * (2 * num_folds + 1),
        [ops.CNOT(q0, q1)],
    )
    assert circuit_folded[0] == expected_circuit_folded_1

    # Second element of list should consist of circuit with only second layer
    # folded.
    expected_circuit_folded_2 = Circuit(
        [ops.H(q0)],
        [ops.CNOT(q0, q1)] * (2 * num_folds + 1),
    )
    assert circuit_folded[1] == expected_circuit_folded_2


def test_bad_layers_to_fold():
    q0, q1 = LineQubit.range(2)
    circuit = Circuit(
        [ops.H(q0)],
        [ops.CNOT(q0, q1)],
    )
    with pytest.raises(
        ValueError,
        match="Length of `layers_to_fold` must be equal to length of circuit.",
    ):
        layer_folding(circuit, [1] * len(circuit) + 1)
