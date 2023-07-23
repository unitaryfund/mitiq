import cirq
import numpy as np
import mitiq
import pytest
from mitiq.shadows.shadows_utils import (
    n_measurements_tomography_bound,
    n_measurements_opts_expectation_bound,
    fidelity,
    transform_to_cirq_paulistring,
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
    observables = [cirq.X, cirq.Y, cirq.Z]
    N, K = n_measurements_opts_expectation_bound(0.5, observables, 0.1)
    assert isinstance(N, int), f"Expected int, got {type(N)}"
    assert isinstance(K, int), f"Expected int, got {type(K)}"


def test_fidelity():
    state_vector = np.array([0.5, 0.5, 0.5, 0.5])
    rho = np.eye(4) / 4
    assert np.isclose(
        fidelity(state_vector, rho), 0.25
    ), f"Expected 0.25, got {fidelity(state_vector, rho)}"


def test_transform_to_cirq_paulistring():
    # Create a mitiq PauliString
    string_pauli = "XYZZ"
    mitiq_pauli = mitiq.PauliString(spec=string_pauli)

    # Convert to cirq PauliString
    cirq_pauli = transform_to_cirq_paulistring(mitiq_pauli)
    assert isinstance(cirq_pauli, cirq.PauliString)
    assert str(cirq_pauli) == "X(q(0))*Y(q(1))*Z(q(2))*Z(q(3))"

    # Convert string to cirq PauliString
    cirq_pauli = transform_to_cirq_paulistring(string_pauli)
    assert isinstance(cirq_pauli, cirq.PauliString)
    assert str(cirq_pauli) == "X(q(0))*Y(q(1))*Z(q(2))*Z(q(3))"

    # Test with a cirq.PauliString
    cirq_pauli = cirq.PauliString({cirq.LineQubit(0): cirq.X})
    cirq_pauli_transformed = transform_to_cirq_paulistring(cirq_pauli)
    assert cirq_pauli == cirq_pauli_transformed

    # Test with an invalid input
    with pytest.raises(ValueError):
        transform_to_cirq_paulistring(123)
