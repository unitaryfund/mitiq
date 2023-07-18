# Copyright (C) Unitary Fund
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

"""Test classical shadow estimation process."""

import cirq
import numpy as np
import pytest


from mitiq import MeasurementResult
from mitiq.interface.mitiq_cirq.cirq_utils import (
    sample_bitstrings as cirq_sample_bitstrings,
)

from mitiq.shadows import execute_with_shadows

# define a fully entangled state
qubits = [cirq.LineQubit(i) for i in range(3)]
num_qubits = len(qubits)
circuit = cirq.Circuit([cirq.H(q) for q in qubits])
circuit.append(cirq.CNOT(qubits[0], qubits[1]))
circuit.append(cirq.CNOT(qubits[1], qubits[2]))
# Define list of observables
observables = [cirq.X(q) for q in qubits]


def executor(
    circuit: cirq.Circuit,
) -> MeasurementResult:
    return cirq_sample_bitstrings(
        circuit,
        noise_level=(0,),
        shots=1,
        sampler=cirq.Simulator(),
    )


def test_execute_with_shadows():
    # Define a random circuit

    """Test classical shadow estimation process."""

    # Call the function to test
    result = execute_with_shadows(
        circuit,
        observables=observables,
        state_reconstruction=True,
        num_total_measurements=10,
        executor=executor,
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
    """Test with state_reconstruction set to False and observables provided"""

    result = execute_with_shadows(
        circuit,
        observables=observables,
        state_reconstruction=False,
        num_total_measurements=10,
        k_shadows=2,
        executor=executor,
    )
    assert isinstance(result, dict)
    assert "est_observables" in result


def test_execute_with_shadows_no_state_reconstruction_error_rate():
    """Test with state_reconstruction set to False,
    observables and error_rate provided"""

    result1 = execute_with_shadows(
        circuit,
        observables=observables,
        state_reconstruction=False,
        num_total_measurements=10,
        error_rate=0.9,
        failure_rate=0.01,
        executor=executor,
    )

    result2 = execute_with_shadows(
        circuit,
        observables=observables,
        state_reconstruction=False,
        error_rate=0.9,
        failure_rate=0.01,
        executor=executor,
    )

    assert isinstance(result1, dict)
    assert "est_observables" in result1
    for key in result1.keys():
        assert len(result1[key]) == len(result2[key])


def test_execute_with_shadows_random_seed():
    """Test with different random seeds"""
    result1 = execute_with_shadows(
        circuit,
        observables=observables,
        state_reconstruction=False,
        num_total_measurements=10,
        k_shadows=2,
        random_seed=1,
        executor=executor,
    )
    result2 = execute_with_shadows(
        circuit,
        observables=observables,
        state_reconstruction=False,
        num_total_measurements=10,
        k_shadows=2,
        random_seed=2,
        executor=executor,
    )
    for key in result1.keys():
        assert not np.array_equal(result1[key], result2[key])


def test_execute_with_shadows_no_observables_no_reconstruction():
    """Test with no observables and no state_reconstruction"""

    # observables is None and state_reconstruction is False
    with pytest.raises(AssertionError) as excinfo:
        execute_with_shadows(
            circuit, state_reconstruction=False, executor=executor
        )

    assert (
        str(excinfo.value) == "observables must be provided"
        " if state_reconstruction is False"
    )


def test_execute_with_shadows_error_rate_and_state_reconstruction():
    """Test with error_rate and state_reconstruction set to True"""

    # error_rate is provided and state_reconstruction is True
    result = execute_with_shadows(
        circuit,
        error_rate=0.99,
        state_reconstruction=True,
        executor=executor,
    )

    # Assert that est_density_matrix is present in the output
    assert "est_density_matrix" in result


def test_execute_with_shadows_error_rate_without_failure_rate():
    """Assert that AssertionError is raised when error_rate
    is given without failure_rate"""

    with pytest.raises(AssertionError):
        execute_with_shadows(
            circuit,
            observables=observables,
            error_rate=0.99,
            state_reconstruction=False,
            executor=executor,
        )
