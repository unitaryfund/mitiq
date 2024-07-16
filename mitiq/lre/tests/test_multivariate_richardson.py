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
    sample_matrix,
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
