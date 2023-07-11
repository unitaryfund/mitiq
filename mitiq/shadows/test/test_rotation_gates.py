import pytest
import time
import cirq
import numpy as np
import cirq.testing

from mitiq.shadows.rotation_gates import (
    generate_random_pauli_strings,
    get_rotated_circuits,
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
    # the characters X, Y, and Z
    # print(pauli_strings)
    for s in pauli_strings:
        assert len(s) == num_qubits
        assert set(s).issubset({"X", "Y", "Z"})


def test_get_rotated_circuits():
    """Tests that the circuit is rotated."""
    # Set up the circuit and pauli strings.
    num_qubits = 3
    circuit = cirq.testing.random_circuit(num_qubits, 10, 0.5)
    num_qubits = len(list(circuit.all_qubits()))
    pauli_strings = generate_random_pauli_strings(num_qubits, 5)

    # Rotate the circuit.
    rotated_circuits = get_rotated_circuits(circuit, pauli_strings)

    assert len(rotated_circuits) == len(pauli_strings)
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
            num_strings[i] / num_strings[i - 1], rel=0.5
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
        assert log_ratio_times == pytest.approx(log_ratio_qubits, rel=0.8)
