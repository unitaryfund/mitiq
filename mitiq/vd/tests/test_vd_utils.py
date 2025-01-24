from mitiq.vd.vd_utils import _copy_circuit_parallel
import cirq
import numpy as np


def test_copy_circuit_parallel_lengths():
    circuit = cirq.Circuit(
        cirq.H(cirq.LineQubit(0)), cirq.SWAP(cirq.LineQubit(0), cirq.LineQubit(1))
    )
    for M in range(2, 10):
        new_circuit = _copy_circuit_parallel(circuit, M)
        assert len(new_circuit.all_qubits()) == 2 * M

    circuit = cirq.Circuit(
        cirq.X(cirq.LineQubit(0)), cirq.Y(cirq.LineQubit(1)), cirq.Z(cirq.LineQubit(2))
    )
    for M in range(2, 10):
        new_circuit = _copy_circuit_parallel(circuit, M)
        assert len(new_circuit.all_qubits()) == 3 * M


def test_copy_circuit_parallel_full():

    M = 2
    qubits = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(cirq.H(qubits[0]), cirq.CNOT(qubits[0], qubits[1]))
    new_circuit = _copy_circuit_parallel(circuit, M)

    expected_qubits = cirq.LineQubit.range(2 * M)
    expected_circuit = cirq.Circuit(
        cirq.H(expected_qubits[0]),
        cirq.CNOT(expected_qubits[0], expected_qubits[1]),
        cirq.H(expected_qubits[2]),
        cirq.CNOT(expected_qubits[2], expected_qubits[3]),
    )

    new_unitary = cirq.unitary(new_circuit)
    expected_unitary = cirq.unitary(expected_circuit)
    assert np.allclose(new_unitary, expected_unitary)

    M = 3
    qubits = cirq.LineQubit.range(1)
    circuit = cirq.Circuit(cirq.X(qubits[0]))
    new_circuit = _copy_circuit_parallel(circuit, M)

    expected_qubits = cirq.LineQubit.range(1 * M)
    expected_circuit = cirq.Circuit(
        cirq.X(expected_qubits[0]),
        cirq.X(expected_qubits[1]),
        cirq.X(expected_qubits[2]),
    )

    new_unitary = cirq.unitary(new_circuit)
    expected_unitary = cirq.unitary(expected_circuit)
    assert np.allclose(new_unitary, expected_unitary)
