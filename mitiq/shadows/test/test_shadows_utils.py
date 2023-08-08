# Copyright (C) Unitary Fund
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

"""Defines utility functions for classical shadows protocol."""

import numpy as np
import cirq
import mitiq
from mitiq.shadows.shadows_utils import (
    kronecker_product,
    operator_ptm_vector_rep,
    eigenvalues_to_bitstring,
    bitstring_to_eigenvalues,
    create_string,
    n_measurements_tomography_bound,
    n_measurements_opts_expectation_bound,
    fidelity,
)


# Tests start here
def test_kronecker_product():
    matrices = [np.array([[1, 2], [3, 4]]), np.array([[0, 1], [1, 0]])]
    expected_result = np.array(
        [[0, 1, 0, 2], [1, 0, 2, 0], [0, 3, 0, 4], [3, 0, 4, 0]]
    )
    np.testing.assert_array_equal(kronecker_product(matrices), expected_result)


def test_operator_ptm_vector_rep():
    opt = cirq.I._unitary_() / np.sqrt(2)
    expected_result = np.array([1.0, 0.0, 0.0, 0.0])
    np.testing.assert_array_almost_equal(
        operator_ptm_vector_rep(opt), expected_result
    )


def test_eigenvalues_to_bitstring():
    values = [-1, 1, 1]
    assert eigenvalues_to_bitstring(values) == "100"
    assert bitstring_to_eigenvalues(eigenvalues_to_bitstring(values)) == values


def test_bitstring_to_eigenvalues():
    bitstring = "100"
    np.testing.assert_array_equal(
        bitstring_to_eigenvalues(bitstring), np.array([-1, 1, 1])
    )
    assert (
        eigenvalues_to_bitstring(bitstring_to_eigenvalues(bitstring))
        == bitstring
    )


def test_create_string():
    str_len = 5
    loc_list = [1, 3]
    assert create_string(str_len, loc_list) == "01010"


def test_n_measurements_tomography_bound():
    assert (
        n_measurements_tomography_bound(0.5, 2) == 2176
    ), f"Expected 2176, got {n_measurements_tomography_bound(0.5, 2)}"
    assert (
        n_measurements_tomography_bound(1.0, 1) == 136
    ), f"Expected 136, got {n_measurements_tomography_bound(1.0, 1)}"
    assert (
        n_measurements_tomography_bound(0.1, 3) == 217599
    ), f"Expected 217599, got {n_measurements_tomography_bound(0.1, 3)}"


def test_n_measurements_opts_expectation_bound():
    observables = [
        mitiq.PauliString("X"),
        mitiq.PauliString("Y"),
        mitiq.PauliString("Z"),
    ]
    N, K = n_measurements_opts_expectation_bound(0.5, observables, 0.1)
    assert isinstance(N, int), f"Expected int, got {type(N)}"
    assert isinstance(K, int), f"Expected int, got {type(K)}"


def test_fidelity():
    state_vector = np.array([0.5, 0.5, 0.5, 0.5])
    rho = np.eye(4) / 4
    assert np.isclose(
        fidelity(state_vector, rho), 0.25
    ), f"Expected 0.25, got {fidelity(state_vector, rho)}"
