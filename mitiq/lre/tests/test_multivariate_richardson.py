# Copyright (C) Unitary Fund
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for multivariate extrapolation inference functions."""

from math import comb

import numpy as np
import pytest
from cirq import Circuit, LineQubit, ops
from sympy import Symbol

from mitiq.lre.inference.multivariate_richardson import (
    full_monomial_basis_terms,
    linear_combination_coefficients,
    sample_matrix,
)
from mitiq.lre.multivariate_scaling.layerwise_folding import (
    _get_num_layers_without_measurements,
    _get_scale_factor_vectors,
    multivariate_layer_scaling,
)

qreg1 = LineQubit.range(3)
test_circuit1 = Circuit(
    [ops.H.on_each(*qreg1)],
    [ops.CNOT.on(qreg1[0], qreg1[1])],
)
test_circuit2 = Circuit(
    [ops.H.on_each(*qreg1)],
    [ops.CNOT.on(qreg1[0], qreg1[1])],
    [ops.X.on(qreg1[2])],
    [ops.TOFFOLI.on(*qreg1)],
)


@pytest.mark.parametrize(
    "test_num_layers, test_degree, expected_basis",
    [
        (1, 1, [1, Symbol("λ_1")]),
        (
            2,
            2,
            [
                1,
                Symbol("λ_2"),
                Symbol("λ_1"),
                Symbol("λ_2") ** 2,
                Symbol("λ_1") * Symbol("λ_2"),
                Symbol("λ_1") ** 2,
            ],
        ),
        (
            3,
            2,
            [
                1,
                Symbol("λ_3"),
                Symbol("λ_2"),
                Symbol("λ_1"),
                Symbol("λ_3") ** 2,
                Symbol("λ_2") * Symbol("λ_3"),
                Symbol("λ_2") ** 2,
                Symbol("λ_1") * Symbol("λ_3"),
                Symbol("λ_1") * Symbol("λ_2"),
                Symbol("λ_1") ** 2,
            ],
        ),
    ],
)
def test_full_monomial_basis(test_num_layers, test_degree, expected_basis):
    calculated_basis = full_monomial_basis_terms(test_num_layers, test_degree)
    assert len(calculated_basis) == comb(
        test_num_layers + test_degree, test_degree
    )

    assert calculated_basis == expected_basis


@pytest.mark.parametrize(
    "test_circ, test_degree, expected_matrix",
    [
        (
            test_circuit1,
            2,
            np.array(
                [
                    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 3.0, 1.0, 3.0, 9.0],
                    [1.0, 3.0, 1.0, 9.0, 3.0, 1.0],
                    [1.0, 1.0, 5.0, 1.0, 5.0, 25.0],
                    [1.0, 3.0, 3.0, 9.0, 9.0, 9.0],
                    [1.0, 5.0, 1.0, 25.0, 5.0, 1.0],
                ]
            ),
        ),
        (
            test_circuit2,
            2,
            np.array(
                [
                    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 3.0, 1.0, 1.0, 1.0, 3.0, 3.0, 9.0],
                    [1.0, 1.0, 3.0, 1.0, 1.0, 3.0, 9.0, 1.0, 3.0, 1.0],
                    [1.0, 3.0, 1.0, 1.0, 9.0, 3.0, 1.0, 3.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 5.0, 1.0, 1.0, 1.0, 5.0, 5.0, 25.0],
                    [1.0, 1.0, 3.0, 3.0, 1.0, 3.0, 9.0, 3.0, 9.0, 9.0],
                    [1.0, 3.0, 1.0, 3.0, 9.0, 3.0, 1.0, 9.0, 3.0, 9.0],
                    [1.0, 1.0, 5.0, 1.0, 1.0, 5.0, 25.0, 1.0, 5.0, 1.0],
                    [1.0, 3.0, 3.0, 1.0, 9.0, 9.0, 9.0, 3.0, 3.0, 1.0],
                    [1.0, 5.0, 1.0, 1.0, 25.0, 5.0, 1.0, 5.0, 1.0, 1.0],
                ]
            ),
        ),
    ],
)
def test_sample_matrix(test_circ, test_degree, expected_matrix):
    assert (
        expected_matrix - sample_matrix(test_circ, test_degree, 1)
    ).all() <= 1e-3


@pytest.mark.parametrize(
    "test_circ, test_degree, test_fold_multiplier, expected_matrix",
    [
        (
            test_circuit1,
            2,
            3,
            [
                1.5555555555555578,
                -0.38888888888888934,
                -0.38888888888888934,
                0.09722222222222215,
                0.027777777777777804,
                0.09722222222222232,
            ],
        ),
        (
            test_circuit2,
            2,
            2,
            [
                2.4062499999999956,
                -0.6874999999999987,
                -0.6874999999999987,
                -0.6874999999999987,
                0.15624999999999956,
                0.06249999999999997,
                0.06249999999999997,
                0.15624999999999956,
                0.06249999999999997,
                0.15624999999999956,
            ],
        ),
    ],
)
def test_coeffs(test_circ, test_degree, test_fold_multiplier, expected_matrix):
    assert (
        abs(
            np.array(expected_matrix)
            - np.array(
                linear_combination_coefficients(
                    test_circ, test_degree, test_fold_multiplier
                )
            )
        ).all()
        <= 1e-3
    )


@pytest.mark.parametrize(
    "test_input, test_degree, test_fold_multiplier, error_msg",
    [
        (
            test_circuit1,
            0,
            1,
            "Multinomial degree must be greater than or equal to 1.",
        ),
        (
            test_circuit1,
            1,
            0,
            "Fold multiplier must be greater than or equal to 1.",
        ),
    ],
)
def test_invalid_degree_fold_multiplier_sample_matrix(
    test_input, test_degree, test_fold_multiplier, error_msg
):
    """Ensures that the args for the sample matrix
    an error for an invalid value."""
    with pytest.raises(ValueError, match=error_msg):
        sample_matrix(test_input, test_degree, test_fold_multiplier)


@pytest.mark.parametrize(
    "test_input, degree, test_fold_multiplier",
    [
        (test_circuit1, 1, 1),
        (test_circuit1, 2, 1),
        (test_circuit1, 3, 5),
        (test_circuit1, 4, 7),
        (test_circuit1, 2, 2),
        (test_circuit1, 2, 3),
    ],
)
def test_square_sample_matrix(test_input, degree, test_fold_multiplier):
    """Check if the sample matrix will always be a square.

    The terms in the monomial basis define the total rows of the sample matrix
    & the generated scale factors for a fold multiplier define the number of
    columns.
    """
    num_layers = _get_num_layers_without_measurements(test_input)
    calculated_basis = full_monomial_basis_terms(num_layers, degree)
    calculated_scale_factor_vectors = _get_scale_factor_vectors(
        test_input, degree, test_fold_multiplier
    )
    assert len(calculated_basis) == len(calculated_scale_factor_vectors)


def test_lre_inference_with_chunking():
    circ = test_circuit1 * 7
    chunked_sample_matrix_dim = sample_matrix(circ, 2, 2, 4).shape
    non_chunked_sample_matrix_dim = sample_matrix(circ, 2, 2).shape
    assert chunked_sample_matrix_dim[0] < non_chunked_sample_matrix_dim[0]


def test_sample_matrix_numerical_stability():
    large_circuit = Circuit([ops.H.on(LineQubit(i)) for i in range(10000)])
    matrix = sample_matrix(large_circuit, 5, 10000)
    assert np.isfinite(matrix).all()
    assert not np.isnan(matrix).any()


@pytest.mark.parametrize("num_chunks", [None, 2, 3])
def test_eval(num_chunks):
    coeffs = linear_combination_coefficients(
        7 * test_circuit2, 2, 2, num_chunks
    )
    multiple_scaled_circuits = multivariate_layer_scaling(
        7 * test_circuit2, 2, 2, num_chunks
    )
    assert len(coeffs) == len(multiple_scaled_circuits)
    assert np.isclose(sum(coeffs), 1.0)  # Coefficients should sum to 1
