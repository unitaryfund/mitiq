import timeit

import cirq

from mitiq.shadows.rotation_gates import (
    generate_random_pauli_strings,
    get_rotated_circuits,
)


def test_generate_random_pauli_strings():
    num_qubits = 5
    num_strings = 10
    pauli_strings = generate_random_pauli_strings(num_qubits, num_strings)

    assert isinstance(pauli_strings, list)
    assert len(pauli_strings) == num_strings
    for s in pauli_strings:
        assert len(s) == num_qubits
        assert set(s).issubset({"X", "Y", "Z"})


def test_get_rotated_circuits():
    num_qubits = 3
    circuit = cirq.testing.random_circuit(num_qubits, 10, 0.5)
    pauli_strings = generate_random_pauli_strings(num_qubits, 5)

    rotated_circuits = get_rotated_circuits(circuit, pauli_strings)

    assert len(rotated_circuits) == len(pauli_strings)
    for rc in rotated_circuits:
        assert isinstance(rc, cirq.Circuit)


def test_get_rotated_circuits_time():
    qubit = cirq.LineQubit.range(5)
    circuit = cirq.Circuit(cirq.H(q) for q in qubit)
    pauli_strings = generate_random_pauli_strings(len(qubit), 1000)
    # Measure the execution time for attaching random rotate gates
    execution_time = timeit.timeit(
        lambda: get_rotated_circuits(circuit, pauli_strings), number=1
    )
    # Check if the execution time grows linearly with the number of strings
    assert (
        execution_time < 10
    )  # Adjust this threshold based on your system performance
