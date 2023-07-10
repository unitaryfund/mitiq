import time

import cirq
import pytest

from mitiq import Executor
from mitiq.shadows.computational_basis_measurement import (
    shadow_measure_with_executor,
)
from mitiq.shadows.executor_functions import cirq_simulator_shadow_executor_fn

executor = Executor(
    cirq_simulator_shadow_executor_fn, max_batch_size=int(1e10)
)
n_total_measurements = 10


def simple_test_circuit(qubits):
    circuit = cirq.Circuit()
    num_qubits = len(qubits)
    for i, qubit in enumerate(qubits):
        circuit.append(cirq.H(qubit))
    for i in range(num_qubits - 1):
        circuit.append(cirq.CNOT(qubits[i], qubits[i + 1]))
    return circuit


@pytest.mark.parametrize("n_qubits", [1, 2, 5])
def test_shadow_measure_with_executor_no_errors(n_qubits: int):
    qubits = cirq.LineQubit.range(n_qubits)
    circuit = simple_test_circuit(qubits)
    shadow_measure_with_executor(circuit, executor, n_total_measurements=10)


@pytest.mark.parametrize("n_qubits", [1, 2, 5])
def test_shadow_measure_with_executor_output_dimensions(n_qubits: int):
    qubits = cirq.LineQubit.range(n_qubits)
    circuit = simple_test_circuit(qubits)
    shadow_outcomes, pauli_strings = shadow_measure_with_executor(
        circuit, executor, n_total_measurements
    )
    assert shadow_outcomes.shape == (
        n_total_measurements,
        n_qubits,
    ), (
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


@pytest.mark.parametrize("n_qubits", [1, 2, 5])
def test_shadow_measure_with_executor_output_types(n_qubits: int):
    qubits = cirq.LineQubit.range(n_qubits)
    circuit = simple_test_circuit(qubits)
    shadow_outcomes, pauli_strings = shadow_measure_with_executor(
        circuit, executor, n_total_measurements
    )
    assert (
        shadow_outcomes[0].dtype == int
    ), f"Shadow outcomes have incorrect dtype, expected int, got {shadow_outcomes.dtype}"
    assert isinstance(
        pauli_strings[0], str
    ), f"Pauli strings have incorrect dtype, expected str, got {pauli_strings.dtype}"


@pytest.mark.parametrize("n_qubits", [1, 2, 5])
def test_shadow_measure_with_executor_time_growth(n_qubits: int):
    qubits = cirq.LineQubit.range(n_qubits)
    circuit = simple_test_circuit(qubits)
    times = []
    measurements = [10, 20, 30, 40, 50]
    for n in measurements:
        start_time = time.time()
        shadow_measure_with_executor(circuit, executor, n)
        times.append(time.time() - start_time)
    for i in range(1, len(times)):
        assert times[i] / times[i - 1] == pytest.approx(
            measurements[i] / measurements[i - 1], rel=0.2
        )
