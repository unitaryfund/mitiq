# Copyright (C) Unitary Fund
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for scaling by layer."""

from cirq import Circuit, LineQubit, ops

from mitiq.zne.scaling import layer_folding


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
        [ops.H(q[0])] * 3,
        [ops.H(q[1])] * 3,
        [ops.CNOT(*q[1:])] * 3,
        ops.measure_each(*q),
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
