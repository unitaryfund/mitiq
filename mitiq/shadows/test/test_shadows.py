# Copyright (C) Unitary Fund
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

"""Test classical shadow estimation process."""

import cirq
import numpy as np

from mitiq.shadows import execute_with_shadows


def test_execute_with_shadows():
    # Define a random circuit
    qubits = [cirq.LineQubit(i) for i in range(3)]
    circuit = cirq.testing.random_circuit(
        qubits=qubits, n_moments=5, op_density=0.7
    )

    # Define observables
    observables = [cirq.X(qubits[i]) for i in range(3)]

    # Call the function to test
    result = execute_with_shadows(
        circuit,
        observables=observables,
        state_reconstruction=True,
        num_total_measurements=10,
    )

    # Check that the result is a dictionary
    assert isinstance(
        result, dict
    ), f"Expected a dictionary, got {type(result)}"

    # Check that the dictionary has the expected keys
    expected_keys = ["shadow_outcomes", "pauli_strings", "est_density_matrix"]
    for key in expected_keys:
        assert key in result, f"Expected key '{key}' in result"


def test_execute_with_shadows_no_state_reconstruction():
    # Test with state_reconstruction set to False and observables provided
    qubits = [cirq.LineQubit(i) for i in range(3)]
    circuit = cirq.testing.random_circuit(
        qubits=qubits, n_moments=5, op_density=0.7
    )

    observables = [cirq.X(q) for q in qubits]
    result = execute_with_shadows(
        circuit,
        observables=observables,
        state_reconstruction=False,
        num_total_measurements=10,
        k_shadows=10,
    )
    assert isinstance(result, dict)
    assert "est_observables" in result


def test_execute_with_shadows_no_state_reconstruction_error_rate():
    # Test with state_reconstruction set to False,
    # observables and error_rate provided
    qubits = [cirq.LineQubit(i) for i in range(3)]
    circuit = cirq.testing.random_circuit(
        qubits=qubits, n_moments=5, op_density=0.7
    )

    observables = [cirq.X(q) for q in qubits]

    result1 = execute_with_shadows(
        circuit,
        observables=observables,
        state_reconstruction=False,
        num_total_measurements=10,
        error_rate=0.9,
        precision=0.99,
    )

    result2 = execute_with_shadows(
        circuit,
        observables=observables,
        state_reconstruction=False,
        error_rate=0.9,
        precision=0.99,
    )

    assert isinstance(result1, dict)
    assert "est_observables" in result1
    for key in result1.keys():
        assert len(result1[key]) == len(result2[key])


def test_execute_with_shadows_random_seed():
    # Test with different random seeds
    qubits = [cirq.LineQubit(i) for i in range(3)]
    circuit = cirq.Circuit([cirq.H(q) for q in qubits])
    observables = [cirq.X(q) for q in qubits]
    result1 = execute_with_shadows(
        circuit,
        observables=observables,
        state_reconstruction=False,
        num_total_measurements=10,
        k_shadows=2,
        random_seed=1,
    )
    result2 = execute_with_shadows(
        circuit,
        observables=observables,
        state_reconstruction=False,
        num_total_measurements=10,
        k_shadows=2,
        random_seed=2,
    )
    for key in result1.keys():
        assert not np.array_equal(result1[key], result2[key])


def test_execute_with_shadows_sampling_function():
    # Test with different sampling functions
    qubits = [cirq.LineQubit(i) for i in range(3)]
    circuit = cirq.testing.random_circuit(
        qubits=qubits, n_moments=5, op_density=0.7
    )
    observables = [cirq.X(q) for q in qubits]
    result1 = execute_with_shadows(
        circuit,
        observables=observables,
        state_reconstruction=False,
        num_total_measurements=10,
        k_shadows=2,
        sampling_function="cirq",
    )
    result2 = execute_with_shadows(
        circuit,
        observables=observables,
        state_reconstruction=False,
        num_total_measurements=10,
        k_shadows=2,
        sampling_function="qiskit",
    )
    for key in result1.keys():
        assert not np.array_equal(result1[key], result2[key])


def test_execute_with_shadows_sampling_function_config():
    # Test with different sampling function configs
    qubits = [cirq.LineQubit(i) for i in range(3)]
    circuit = cirq.testing.random_circuit(
        qubits=qubits, n_moments=5, op_density=0.7
    )
    observables = [cirq.X(q) for q in qubits]
    config1 = {"option1": "value1"}
    config2 = {"option1": "value2"}
    result1 = execute_with_shadows(
        circuit,
        observables=observables,
        state_reconstruction=False,
        num_total_measurements=1000,
        k_shadows=10,
        sampling_function_config=config1,
    )
    result2 = execute_with_shadows(
        circuit,
        observables=observables,
        state_reconstruction=False,
        num_total_measurements=1000,
        k_shadows=10,
        sampling_function_config=config2,
    )
    for key in result1.keys():
        assert not np.array_equal(result1[key], result2[key])
