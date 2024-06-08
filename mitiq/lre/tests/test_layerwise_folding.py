# Copyright (C) Unitary Fund
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for scaling noise by unitary folding of layers in the input
circuit to allow for multivariate extrapolation."""

from copy import deepcopy

import pytest
from cirq import Circuit, LineQubit, ops

from mitiq.lre.multivariate_scaling.layerwise_folding import (
    _get_num_layers_without_measurements,
    _get_scale_factor_vectors,
    multivariate_layer_scaling,
)

qreg1 = LineQubit.range(3)
test_circuit1 = Circuit(
    [ops.H.on_each(*qreg1)],
    [ops.CNOT.on(qreg1[0], qreg1[1])],
    [ops.X.on(qreg1[2])],
    [ops.TOFFOLI.on(*qreg1)],
)

test_circuit1_with_measurements = deepcopy(test_circuit1)
test_circuit1_with_measurements.append(ops.measure_each(*qreg1))


def test_multivariate_layerwise_scaling():
    """Checks if multiple scaled circuits are returned to fit the required
    folding pattern for multivariate extrapolation."""
    multiple_scaled_circuits = multivariate_layer_scaling(
        test_circuit1, 2, 2, 3
    )

    assert len(multiple_scaled_circuits) == 10
    folding_pattern = [
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
    ]

    i = 0
    for scale_factor_vector in folding_pattern:
        scale_layer1, scale_layer2, scale_layer3 = scale_factor_vector
        expected_circuit = Circuit(
            [ops.H.on_each(*qreg1)] * scale_layer1,
            [ops.CNOT.on(qreg1[0], qreg1[1]), ops.X.on(qreg1[2])]
            * scale_layer2,
            [ops.TOFFOLI.on(*qreg1)] * scale_layer3,
        )
        assert expected_circuit == multiple_scaled_circuits[i]
        i += 1


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
def test_get_scale_factor_vectors_no_chunking(
    test_input, degree, test_fold_multiplier, expected_scale_factor_vectors
):
    """Verifies vectors of scale factors are calculated accurately."""
    calculated_scale_factor_vectors = _get_scale_factor_vectors(
        test_input, degree, test_fold_multiplier
    )

    assert calculated_scale_factor_vectors == expected_scale_factor_vectors


@pytest.mark.parametrize(
    "test_input, degree, test_fold_multiplier, test_chunks, expected_size",
    [
        (test_circuit1, 1, 1, 2, 3),
        (test_circuit1, 2, 1, 3, 10),
        (test_circuit1, 2, 3, 2, 6),
    ],
)
def test_get_scale_factor_vectors_with_chunking(
    test_input, degree, test_fold_multiplier, test_chunks, expected_size
):
    """Verifies vectors of scale factors are calculated accurately."""
    calculated_scale_factor_vectors = _get_scale_factor_vectors(
        test_input, degree, test_fold_multiplier, test_chunks
    )

    assert len(calculated_scale_factor_vectors) == expected_size


@pytest.mark.parametrize(
    "test_input, num_chunks, error_msg",
    [
        (test_circuit1, 0, "The number of chunks should be >= to 1."),
        (
            test_circuit1,
            5,
            "Number of chunks > the number of layers in the circuit.",
        ),
    ],
)
def test_invalid_num_chunks(test_input, num_chunks, error_msg):
    """Ensures that the number of intended chunks in the input circuit raises
    an error for an invalid value."""
    with pytest.raises(ValueError, match=error_msg):
        _get_scale_factor_vectors(test_input, 2, 2, num_chunks)


@pytest.mark.parametrize(
    "test_input, test_degree, test_fold_multiplier, error_msg",
    [
        (test_circuit1, 0, 1, "Multinomial degree not >= to 1."),
        (
            test_circuit1,
            1,
            0,
            "Fold multiplier not >= to 1.",
        ),
    ],
)
def test_invalid_degree_fold_multiplier(
    test_input, test_degree, test_fold_multiplier, error_msg
):
    """Ensures that the number of intended chunks in the input circuit raises
    an error for an invalid value."""
    with pytest.raises(ValueError, match=error_msg):
        multivariate_layer_scaling(
            test_input, test_degree, test_fold_multiplier
        )
