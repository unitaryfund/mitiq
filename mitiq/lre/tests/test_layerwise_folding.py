# Copyright (C) Unitary Fund
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for scaling noise by unitary folding of layers in the input
circuit to allow for multivariate extrapolation."""

from copy import deepcopy

import cirq
import pytest

from mitiq.lre.multivariate_scaling.layerwise_folding import (
    _get_num_layers_without_measurements,
    _get_scale_factor_vectors,
)

qreg1 = cirq.LineQubit.range(3)
test_circuit1 = cirq.Circuit(
    [cirq.ops.H.on_each(*qreg1)],
    [cirq.ops.CNOT.on(qreg1[0], qreg1[1])],
    [cirq.ops.X.on(qreg1[2])],
    [cirq.ops.TOFFOLI.on(*qreg1)],
)

test_circuit1_with_measurements = deepcopy(test_circuit1)
test_circuit1_with_measurements.append(cirq.ops.measure_each(*qreg1))


@pytest.mark.parametrize(
    "test_input, expected",
    [(test_circuit1, 3), (test_circuit1_with_measurements, 3)],
)
def test_get_num_layers(test_input, expected):
    """Verifies function works as expected."""
    calculated_num_layers = _get_num_layers_without_measurements(test_input)

    assert calculated_num_layers == expected


@pytest.mark.parametrize(
    "test_input, degree, test_fold_multiplier, expected_scale_factor_vectors",
    [
        (test_circuit1, 1, 0, [(1, 1, 1), (1, 1, 1), (1, 1, 1), (1, 1, 1)]),
        (test_circuit1, 1, 1, [(1, 1, 1), (3, 1, 1), (1, 3, 1), (1, 1, 3)]),
        (
            test_circuit1,
            2,
            1,
            [
                (1, 1, 1),
                (3, 1, 1),
                (1, 3, 1),
                (1, 1, 3),
                (5, 1, 1),
                (3, 3, 1),
                (3, 1, 3),
                (1, 5, 1),
                (1, 3, 3),
                (1, 1, 5),
            ],
        ),
        (
            test_circuit1,
            2,
            2,
            [
                (1, 1, 1),
                (5, 1, 1),
                (1, 5, 1),
                (1, 1, 5),
                (9, 1, 1),
                (5, 5, 1),
                (5, 1, 5),
                (1, 9, 1),
                (1, 5, 5),
                (1, 1, 9),
            ],
        ),
        (
            test_circuit1,
            2,
            3,
            [
                (1, 1, 1),
                (7, 1, 1),
                (1, 7, 1),
                (1, 1, 7),
                (13, 1, 1),
                (7, 7, 1),
                (7, 1, 7),
                (1, 13, 1),
                (1, 7, 7),
                (1, 1, 13),
            ],
        ),
        (
            test_circuit1_with_measurements,
            1,
            0,
            [(1, 1, 1), (1, 1, 1), (1, 1, 1), (1, 1, 1)],
        ),(
            test_circuit1_with_measurements,
            1,
            1,
            [(1, 1, 1), (3, 1, 1), (1, 3, 1), (1, 1, 3)],
        ),
        (
            test_circuit1_with_measurements,
            2,
            1,
            [
                (1, 1, 1),
                (3, 1, 1),
                (1, 3, 1),
                (1, 1, 3),
                (5, 1, 1),
                (3, 3, 1),
                (3, 1, 3),
                (1, 5, 1),
                (1, 3, 3),
                (1, 1, 5),
            ],
        ),
        (
            test_circuit1_with_measurements,
            2,
            2,
            [
                (1, 1, 1),
                (5, 1, 1),
                (1, 5, 1),
                (1, 1, 5),
                (9, 1, 1),
                (5, 5, 1),
                (5, 1, 5),
                (1, 9, 1),
                (1, 5, 5),
                (1, 1, 9),
            ],
        ),
        (
            test_circuit1_with_measurements,
            2,
            3,
            [
                (1, 1, 1),
                (7, 1, 1),
                (1, 7, 1),
                (1, 1, 7),
                (13, 1, 1),
                (7, 7, 1),
                (7, 1, 7),
                (1, 13, 1),
                (1, 7, 7),
                (1, 1, 13),
            ],
        ),
    ],
)
def test_get_scale_factor_vectors(
    test_input, degree, test_fold_multiplier, expected_scale_factor_vectors
):
    """Verifies vectors of scale factors are calculated accurately."""
    calculated_scale_factor_vectors = _get_scale_factor_vectors(
        test_input, degree, test_fold_multiplier
    )

    assert calculated_scale_factor_vectors == expected_scale_factor_vectors
