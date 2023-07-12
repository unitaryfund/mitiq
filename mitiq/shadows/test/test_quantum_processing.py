import time

import cirq
import cirq.testing
import numpy as np
import pytest
from qiskit_aer import Aer

from mitiq import MeasurementResult
from mitiq.interface.mitiq_qiskit.conversions import to_qiskit
from mitiq.interface.mitiq_qiskit.qiskit_utils import (
    sample_bitstrings as qiskit_sample_bitstrings,
)
from mitiq.shadows.quantum_processing import (
    generate_random_pauli_strings,
    get_rotated_circuits,
    random_pauli_basis_measurement,
)


def test_generate_random_pauli_strings():
    """
    Test that generate_random_pauli_strings returns a list of Pauli strings
    of the correct length.
    """

    # Define the number of qubits and Pauli strings to generate
    num_qubits = 5
    num_strings = 10

    # Generate the Pauli strings
    pauli_strings = generate_random_pauli_strings(num_qubits, num_strings)

    # Check that the function returned a list of the correct length
    assert isinstance(pauli_strings, list)
    assert len(pauli_strings) == num_strings

    # Check that each Pauli string is the correct length and contains only

    for s in pauli_strings:
        assert len(s) == num_qubits
        assert set(s).issubset({"X", "Y", "Z"})


def test_get_rotated_circuits():
    """Tests that the circuit is rotated."""

    # define circuit a two qubit bell state
    circuit = cirq.Circuit()
    qubits = cirq.LineQubit.range(2)
    circuit.append(cirq.H(qubits[0]))
    circuit.append(cirq.CNOT(qubits[0], qubits[1]))

    # define the pauli measurements to be performed on the circuit
    pauli_strings = ["XY", "YZ"]
    # Rotate the circuit.
    rotated_circuits = get_rotated_circuits(circuit, pauli_strings)
    # Verify that the circuit was rotated.
    circuit_1 = circuit.copy()
    circuit_1.append(cirq.H(qubits[0]))
    circuit_1.append(cirq.S(qubits[1]) ** -1)
    circuit_1.append(cirq.H(qubits[1]))
    circuit_1.append(cirq.measure(*qubits))
    circuit_2 = circuit.copy()
    circuit_2.append(cirq.S(qubits[0]) ** -1)
    circuit_2.append(cirq.H(qubits[0]))
    circuit_2.append(cirq.measure(*qubits))
    assert rotated_circuits == [
        circuit_1,
        circuit_2,
    ], f"Expected {rotated_circuits}, got {[circuit_1, circuit_2]}"
    for rc in rotated_circuits:
        assert isinstance(rc, cirq.Circuit)


def test_generate_random_pauli_strings_time() -> None:
    """
    Test if the execution time of generate_random_pauli_strings linearly with
    the number of Pauli strings.
    """
    # Define the number of qubits
    num_qubits: int = 3
    times: list = []
    num_strings = [100, 200, 300, 400, 500]
    for n in num_strings:
        # Measure the execution time for generating random Pauli strings
        start_time = time.time()
        generate_random_pauli_strings(num_qubits, n)
        times.append(time.time() - start_time)
    for i in range(1, len(times)):
        assert times[i] / times[i - 1] == pytest.approx(
            num_strings[i] / num_strings[i - 1], rel=1
        )


def test_generate_random_pauli_strings_time_power_law() -> None:
    """
    Test if the execution time of generate_random_pauli_strings grows
    as a power law with respect to the number of qubits.
    """
    num_strings: int = 1000
    times: list = []
    num_qubits_list = [3, 6, 9, 12, 15]
    for num_qubits in num_qubits_list:
        # Measure the execution time for generating random Pauli strings
        start_time = time.time()
        generate_random_pauli_strings(num_qubits, num_strings)
        times.append(time.time() - start_time)
    for i in range(1, len(times)):
        log_ratio_times = np.log(times[i] / times[i - 1])
        log_ratio_qubits = np.log(num_qubits_list[i] / num_qubits_list[i - 1])
        assert log_ratio_times == pytest.approx(log_ratio_qubits, rel=2)


n_total_measurements = 10


def simple_test_circuit(qubits):
    circuit = cirq.Circuit()
    num_qubits = len(qubits)
    for i, qubit in enumerate(qubits):
        circuit.append(cirq.H(qubit))
    for i in range(num_qubits - 1):
        circuit.append(cirq.CNOT(qubits[i], qubits[i + 1]))
    return circuit


# test different generators
@pytest.fixture
def sampling_function(request):
    return request.param


@pytest.mark.parametrize(
    "sampling_function",
    ["cirq", "qiskit"],
    indirect=True,
)
@pytest.mark.parametrize("n_qubits", [1, 2, 5])
def test_get_z_basis_measurement_no_errors(
    n_qubits: int, sampling_function: str
):
    qubits = cirq.LineQubit.range(n_qubits)
    circuit = simple_test_circuit(qubits)
    random_pauli_basis_measurement(
        circuit, n_total_measurements=10, sampling_function=sampling_function
    )


@pytest.mark.parametrize(
    "sampling_function",
    ["cirq", "qiskit"],
    indirect=True,
)
@pytest.mark.parametrize("n_qubits", [1, 2, 5])
def test_get_z_basis_measurement_output_dimensions(
    n_qubits: int, sampling_function: str
):
    qubits = cirq.LineQubit.range(n_qubits)
    circuit = simple_test_circuit(qubits)
    shadow_outcomes, pauli_strings = random_pauli_basis_measurement(
        circuit, n_total_measurements, sampling_function=sampling_function
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
    "sampling_function",
    ["cirq", "qiskit"],
    indirect=True,
)
@pytest.mark.parametrize("n_qubits", [1, 2, 5])
def test_get_z_basis_measurement_output_types(
    n_qubits: int, sampling_function: str
):
    qubits = cirq.LineQubit.range(n_qubits)
    circuit = simple_test_circuit(qubits)
    shadow_outcomes, pauli_strings = random_pauli_basis_measurement(
        circuit, n_total_measurements, sampling_function=sampling_function
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
    "sampling_function",
    ["cirq", "qiskit"],
    indirect=True,
)
@pytest.mark.parametrize("n_qubits", [1, 2, 5])
def test_get_z_basis_measurement_time_growth(
    n_qubits: int, sampling_function: str
):
    qubits = cirq.LineQubit.range(n_qubits)
    circuit = simple_test_circuit(qubits)
    times = []
    measurements = [10, 20, 30, 40, 50]
    for n in measurements:
        start_time = time.time()
        random_pauli_basis_measurement(
            circuit, n, sampling_function=sampling_function
        )
        times.append(time.time() - start_time)
    for i in range(1, len(times)):
        assert times[i] / times[i - 1] == pytest.approx(
            measurements[i] / measurements[i - 1], rel=3
        )


def test_user_sampling_bitstrings_fn():
    def customized_fn(
        circuit: cirq.Circuit,
    ) -> MeasurementResult:
        return qiskit_sample_bitstrings(
            to_qiskit(circuit),
            noise_model=None,
            backend=Aer.get_backend("aer_simulator"),
            shots=1,
            measure_all=False,
        )

    qubits = cirq.LineQubit.range(3)
    circuit = simple_test_circuit(qubits)
    shadow_outcomes, pauli_strings = random_pauli_basis_measurement(
        circuit, n_total_measurements, sampling_function=customized_fn
    )
    assert shadow_outcomes.shape == (n_total_measurements, 3,), (
        f"Shadow outcomes have incorrect shape, expected "
        f"{(n_total_measurements, 3)}, got {shadow_outcomes.shape}"
    )
    assert pauli_strings.shape == (n_total_measurements,), (
        f"Pauli strings have incorrect shape, expected "
        f"{(n_total_measurements, 3)}, got {pauli_strings.shape}"
    )
    assert len(pauli_strings[0]) == 3, (
        f"Pauli strings have incorrect number of characters, "
        f"expected {3}, got {len(pauli_strings[0])}"
    )
