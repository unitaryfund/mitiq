# Copyright (C) Unitary Fund
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for quantum processing functions for classical shadows."""

import importlib
import time
from typing import Callable
from unittest.mock import patch

import cirq
import cirq.testing
import numpy as np
import pytest
from qiskit_aer import Aer

import mitiq
from mitiq import MeasurementResult
from mitiq.interface.mitiq_cirq.cirq_utils import (
    sample_bitstrings as cirq_sample_bitstrings,
)
from mitiq.interface.mitiq_qiskit.conversions import to_qiskit
from mitiq.interface.mitiq_qiskit.qiskit_utils import (
    sample_bitstrings as qiskit_sample_bitstrings,
)
from mitiq.shadows.quantum_processing import (
    generate_random_pauli_strings,
    get_rotated_circuits,
    random_pauli_measurement,
)


def test_tqdm_import_available():
    # Test the case where tqdm is available
    import tqdm as tqdm_orig

    assert tqdm_orig is not None
    assert mitiq.shadows.quantum_processing.tqdm is not None


def test_tqdm_import_not_available():
    with patch.dict("sys.modules", {"tqdm.auto": None}):
        importlib.reload(
            mitiq.shadows.quantum_processing
        )  # Reload the module to trigger the import
        assert mitiq.shadows.quantum_processing.tqdm is None


def cirq_executor(
    circuit: cirq.Circuit,
) -> MeasurementResult:
    return cirq_sample_bitstrings(
        circuit,
        noise_level=(0,),
        shots=1,
        sampler=cirq.Simulator(),
    )


def qiskit_executor(
    circuit: cirq.Circuit,
) -> MeasurementResult:
    return qiskit_sample_bitstrings(
        to_qiskit(circuit),
        noise_model=None,
        backend=Aer.get_backend("aer_simulator"),
        shots=1,
        measure_all=False,
    )


def test_generate_random_pauli_strings_time() -> None:
    """
    Test if the execution time of generate_random_pauli_strings linearly with
    the number of Pauli strings.
    """
    # Define the number of qubits
    num_qubits: int = 300
    times: list = []
    num_strings = [3000, 4000, 5000]
    for n in num_strings:
        # Measure the execution time for generating random Pauli strings
        start_time = time.time()
        generate_random_pauli_strings(num_qubits, n)
        times.append(time.time() - start_time)
    for i in range(1, len(times)):
        assert times[i] / times[i - 1] == pytest.approx(
            num_strings[i] / num_strings[i - 1], rel=1
        )


def test_get_rotated_circuits():
    """Tests that the circuit is rotated."""

    # define circuit
    circuit = cirq.Circuit()
    qubits = cirq.LineQubit.range(4)
    circuit.append(cirq.H(qubits[0]))
    circuit.append(cirq.CNOT(qubits[0], qubits[1]))
    circuit.append(cirq.CNOT(qubits[1], qubits[2]))
    circuit.append(cirq.CNOT(qubits[2], qubits[3]))

    # define the pauli measurements to be performed on the circuit
    pauli_strings = ["XYZX"]
    # Rotate the circuit.
    rotated_circuits = get_rotated_circuits(circuit, pauli_strings)
    # Verify that the circuit was rotated.
    circuit_1 = circuit.copy()
    circuit_1.append(cirq.H(qubits[0]))
    circuit_1.append(cirq.S(qubits[1]) ** -1)
    circuit_1.append(cirq.H(qubits[1]))
    circuit_1.append(cirq.H(qubits[3]))
    circuit_1.append(cirq.measure(*qubits))

    assert rotated_circuits[0] == circuit_1

    for rc in rotated_circuits:
        assert isinstance(rc, cirq.Circuit)


# define a simple test circuit for the following tests
def simple_test_circuit(qubits):
    circuit = cirq.Circuit()
    num_qubits = len(qubits)
    circuit.append(cirq.H.on_each(*qubits))
    for i in range(num_qubits - 1):
        circuit.append(cirq.CNOT(qubits[i], qubits[i + 1]))
    return circuit


# test different generators
@pytest.fixture
def executor(request):
    return request.param


@pytest.mark.parametrize(
    "executor",
    [cirq_executor, qiskit_executor],
    indirect=True,
)
@pytest.mark.parametrize("n_qubits", [1, 2, 5])
def test_random_pauli_measurement_no_errors(n_qubits, executor):
    """Test that random_pauli_measurement runs without errors."""
    qubits = cirq.LineQubit.range(n_qubits)
    circuit = simple_test_circuit(qubits)
    random_pauli_measurement(
        circuit, n_total_measurements=10, executor=executor
    )


@pytest.mark.parametrize(
    "executor",
    [cirq_executor, qiskit_executor],
    indirect=True,
)
@pytest.mark.parametrize("n_qubits", [1, 2, 5])
def test_random_pauli_measurement_output_dimensions(
    n_qubits: int, executor: Callable
):
    """Test that random_pauli_measurement returns the correct output
    dimensions."""
    qubits = cirq.LineQubit.range(n_qubits)
    circuit = simple_test_circuit(qubits)
    n_total_measurements = 10
    shadow_outcomes, pauli_strings = random_pauli_measurement(
        circuit, n_total_measurements, executor=executor
    )
    assert shadow_outcomes.shape == (n_total_measurements, n_qubits,), (
        f"Shadow outcomes have incorrect shape, expected "
        f"{(n_total_measurements, n_qubits)}, got {shadow_outcomes.shape}"
    )
    assert pauli_strings.shape == (n_total_measurements,), (
        f"Pauli strings have incorrect shape, expected "
        f"{(n_total_measurements, n_qubits)}, got {pauli_strings.shape}"
    )
    assert len(pauli_strings[0]) == n_qubits, (
        f"Pauli strings have incorrect number of characters, "
        f"expected {n_qubits}, got {len(pauli_strings[0])}"
    )


@pytest.mark.parametrize(
    "executor",
    [cirq_executor, qiskit_executor],
    indirect=True,
)
@pytest.mark.parametrize("n_qubits", [1, 2, 5])
def test_random_pauli_measurement_output_types(
    n_qubits: int, executor: Callable
):
    """Test that random_pauli_measurement returns the correct output types."""
    qubits = cirq.LineQubit.range(n_qubits)
    circuit = simple_test_circuit(qubits)
    shadow_outcomes, pauli_strings = random_pauli_measurement(
        circuit, n_total_measurements=10, executor=executor
    )
    assert shadow_outcomes[0].dtype == int, (
        f"Shadow outcomes have incorrect dtype, expected int, "
        f"got {shadow_outcomes.dtype}"
    )
    assert isinstance(pauli_strings[0], str), (
        f"Pauli strings have incorrect dtype, expected str, "
        f"got {pauli_strings.dtype}"
    )


@pytest.mark.parametrize(
    "executor",
    [cirq_executor, qiskit_executor],
    indirect=True,
)
def test_random_pauli_measurement_time_growth(executor: Callable):
    """Test that random_pauli_measurement scales linearly with the
    number of measurements."""
    n_qubits = 5
    qubits = cirq.LineQubit.range(n_qubits)
    circuit = simple_test_circuit(qubits)
    times = []
    measurements = [10, 20, 30, 40, 50]
    for n in measurements:
        start_time = time.time()
        random_pauli_measurement(circuit, n, executor=executor)
        times.append(time.time() - start_time)
    for i in range(1, len(times)):
        assert times[i] / times[i - 1] == pytest.approx(
            measurements[i] / measurements[i - 1], rel=8
        )


@pytest.mark.parametrize(
    "executor",
    [cirq_executor, qiskit_executor],
    indirect=True,
)
def test_random_pauli_measurement_time_power_growth(
    executor: Callable,
):
    """Test that random_pauli_measurement scales follow power law with the
    number of measurements."""
    n_qubits = [3, 6, 9, 12, 15]

    times = []
    for n in n_qubits:
        qubits = cirq.LineQubit.range(n)
        circuit = simple_test_circuit(qubits)
        start_time = time.time()
        random_pauli_measurement(
            circuit,
            100,  # number of total measurements
            executor=executor,
        )

        times.append(time.time() - start_time)
    for i in range(1, len(times)):
        log_ratio_times = np.log(times[i] / times[i - 1])
        log_ratio_qubits = np.log(n_qubits[i] / n_qubits[i - 1])
        assert log_ratio_times == pytest.approx(log_ratio_qubits, rel=5)
