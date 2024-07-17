# Copyright (C) Unitary Fund
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for multivariate extrapolation inference functions."""

from math import comb

import numpy as np
import pytest
from cirq import Circuit, LineQubit, ops

from mitiq.lre.inference.multivariate_richardson import (
    _create_variable_combinations,
    _get_variables,
    full_monomial_basis,
    linear_combination_coefficients,
    sample_matrix,
)
from mitiq.lre.multivariate_scaling.layerwise_folding import (
    _get_num_layers_without_measurements,
    _get_scale_factor_vectors,
)


@pytest.mark.parametrize(
    "test_num_layers, expected_list",
    [
        (2, ["λ_1", "λ_2"]),
        (3, ["λ_1", "λ_2", "λ_3"]),
        (4, ["λ_1", "λ_2", "λ_3", "λ_4"]),
    ],
)
def test_get_variables(test_num_layers, expected_list):
    calculated_variables = _get_variables(test_num_layers)
    assert len(calculated_variables) == test_num_layers
    assert calculated_variables == expected_list


@pytest.mark.parametrize(
    "test_num_layers, test_degree",
    [(2, 2), (3, 2), (4, 2), (2, 3), (3, 3)],
)
def test_create_variable_combinations(test_num_layers, test_degree):
    calculated_variables = _create_variable_combinations(
        test_num_layers, test_degree
    )
    assert len(calculated_variables) == comb(
        test_num_layers + test_degree, test_degree
    )

    # check last element is an empty tuple to account for degree = 0
    assert not calculated_variables[-1]


@pytest.mark.parametrize(
    "test_num_layers, test_degree, expected_basis",
    [
        (2, 2, ["1", "λ_2", "λ_1", "λ_2**2", "λ_1*λ_2", "λ_1**2"]),
        (
            3,
            2,
            [
                "1",
                "λ_3",
                "λ_2",
                "λ_1",
                "λ_3**2",
                "λ_2*λ_3",
                "λ_2**2",
                "λ_1*λ_3",
                "λ_1*λ_2",
                "λ_1**2",
            ],
        ),
    ],
)
def test_full_monomial_basis(test_num_layers, test_degree, expected_basis):
    calculated_basis = full_monomial_basis(test_num_layers, test_degree)
    assert len(calculated_basis) == comb(
        test_num_layers + test_degree, test_degree
    )

    assert calculated_basis == expected_basis


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
    assert (expected_matrix == sample_matrix(test_circ, test_degree, 1)).all()


@pytest.mark.parametrize(
    "test_circ, test_degree, test_fold_multiplier, expected_matrix",
    [
        (
            test_circuit1,
            2,
            3,
            [
                0.013888888888888876,
                -0.027777777777777804,
                0.0,
                0.013888888888888876,
                0.0,
                0.0,
            ],
        ),
        (
            test_circuit2,
            2,
            2,
            [
                0.03124999999999993,
                -0.06249999999999997,
                0.0,
                0.0,
                0.03124999999999993,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
        ),
    ],
)
def test_coeffs(test_circ, test_degree, test_fold_multiplier, expected_matrix):
    assert expected_matrix == linear_combination_coefficients(
        test_circ, test_degree, test_fold_multiplier
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
    & the generated scale factors for some fold multiplier define the number of
    columns.
    """
    num_layers = _get_num_layers_without_measurements(test_input)
    calculated_basis = full_monomial_basis(num_layers, degree)
    calculated_scale_factor_vectors = _get_scale_factor_vectors(
        test_input, degree, test_fold_multiplier
    )
    assert len(calculated_basis) == len(calculated_scale_factor_vectors)
