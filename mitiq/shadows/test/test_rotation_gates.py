import timeit

import cirq

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
    for s in pauli_strings:
        assert len(s) == num_qubits
        assert set(s).issubset({"X", "Y", "Z"})


def test_get_rotated_circuits():
    """Tests that the circuit is rotated."""
    # Set up the circuit and pauli strings.
    num_qubits = 3
    circuit = cirq.testing.random_circuit(num_qubits, 10, 0.5)
    pauli_strings = generate_random_pauli_strings(num_qubits, 5)

    # Rotate the circuit.
    rotated_circuits = get_rotated_circuits(circuit, pauli_strings)

    # Verify that the circuit was rotated.
    assert len(rotated_circuits) == len(pauli_strings)
    for rc in rotated_circuits:
        assert isinstance(rc, cirq.Circuit)


def test_get_rotated_circuits_time():
    """
    Test if the execution time of get_rotated_circuits grows 
    linearly with the number of Pauli strings.
    """
    qubits = cirq.LineQubit.range(5)
    circuit = cirq.Circuit(cirq.H(q) for q in qubits)
    num_strings = 1000
    pauli_strings = generate_random_pauli_strings(len(qubits), num_strings)
    # Measure the execution time for attaching random rotate gates
    execution_time = timeit.timeit(
        lambda: get_rotated_circuits(circuit, pauli_strings), number=1
    )
    # Check if the execution time grows linearly with the number of strings
    assert (
        execution_time < 10
    )  