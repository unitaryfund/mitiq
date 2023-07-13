# Copyright (C) Unitary Fund
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

"""Tests for utility functions for classical shadows protocol."""
import cirq
import numpy as np
from mitiq.shadows.shadows_utils import (
    min_n_total_measurements,
    calculate_shadow_bound,
    operator_2_norm,
    fidelity,
)


def test_min_n_total_measurements():
    assert (
        min_n_total_measurements(0.5, 2) == 2176
    ), f"Expected 2176, got {min_n_total_measurements(0.5, 2)}"
    assert (
        min_n_total_measurements(1.0, 1) == 136
    ), f"Expected 136, got {min_n_total_measurements(1.0, 1)}"
    assert (
        min_n_total_measurements(0.1, 3) == 217599
    ), f"Expected 217599, got {min_n_total_measurements(0.1, 3)}"


def test_calculate_shadow_bound():
    observables = [cirq.X, cirq.Y, cirq.Z]
    N, K = calculate_shadow_bound(0.5, observables, 0.1)
    assert isinstance(N, int), f"Expected int, got {type(N)}"
    assert isinstance(K, int), f"Expected int, got {type(K)}"


def test_operator_2_norm():
    R = np.array([[1, 2], [3, 4]])
    assert np.isclose(
        operator_2_norm(R), np.sqrt(30)
    ), f"Expected {np.sqrt(30)}, got {operator_2_norm(R)}"


def test_fidelity():
    state_vector = np.array([0.5, 0.5, 0.5, 0.5])
    rho = np.eye(4) / 4
    assert np.isclose(
        fidelity(state_vector, rho), 0.25
    ), f"Expected 0.25, got {fidelity(state_vector, rho)}"
