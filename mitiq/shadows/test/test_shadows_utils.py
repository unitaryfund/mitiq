import numpy as np

import mitiq
from mitiq.shadows.shadows_utils import (
    n_measurements_tomography_bound,
    n_measurements_opts_expectation_bound,
    fidelity,
)


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
