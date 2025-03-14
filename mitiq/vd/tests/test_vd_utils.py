import cirq
import numpy as np

from mitiq.vd.vd_utils import (
    _apply_diagonalizing_gate,
    _copy_circuit_parallel,
    _generate_diagonalizing_gate,
)


def test_copy_circuit_parallel_lengths():
    circuit = cirq.Circuit(
        cirq.H(cirq.LineQubit(0)),
        cirq.SWAP(cirq.LineQubit(0), cirq.LineQubit(1)),
    )
    for num in range(2, 10):
        new_circuit = _copy_circuit_parallel(circuit, num)
        assert len(new_circuit.all_qubits()) == 2 * num

    circuit = cirq.Circuit(
        cirq.X(cirq.LineQubit(0)),
        cirq.Y(cirq.LineQubit(1)),
        cirq.Z(cirq.LineQubit(2)),
    )
    for num in range(2, 10):
        new_circuit = _copy_circuit_parallel(circuit, num)
        assert len(new_circuit.all_qubits()) == 3 * num


def test_copy_circuit_parallel_full():
    num = 2
    qubits = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(cirq.H(qubits[0]), cirq.CNOT(qubits[0], qubits[1]))
    new_circuit = _copy_circuit_parallel(circuit, num)

    expected_qubits = cirq.LineQubit.range(2 * num)
    expected_circuit = cirq.Circuit(
        cirq.H(expected_qubits[0]),
        cirq.CNOT(expected_qubits[0], expected_qubits[1]),
        cirq.H(expected_qubits[2]),
        cirq.CNOT(expected_qubits[2], expected_qubits[3]),
    )

    new_unitary = cirq.unitary(new_circuit)
    expected_unitary = cirq.unitary(expected_circuit)
    assert np.allclose(new_unitary, expected_unitary)

    num = 3
    qubits = cirq.LineQubit.range(1)
    circuit = cirq.Circuit(cirq.X(qubits[0]))
    new_circuit = _copy_circuit_parallel(circuit, num)

    expected_qubits = cirq.LineQubit.range(1 * num)
    expected_circuit = cirq.Circuit(
        cirq.X(expected_qubits[0]),
        cirq.X(expected_qubits[1]),
        cirq.X(expected_qubits[2]),
    )

    new_unitary = cirq.unitary(new_circuit)
    expected_unitary = cirq.unitary(expected_circuit)
    assert np.allclose(new_unitary, expected_unitary)


def test_copy_circuit_parallel_gridqubits():
    num = 3
    qubits = cirq.GridQubit.rect(2, 2)
    circuit = cirq.Circuit(
        cirq.H(qubits[0]),
        cirq.CNOT(qubits[0], qubits[1]),
        cirq.CNOT(qubits[0], qubits[2]),
    )

    new_circuit = _copy_circuit_parallel(circuit, num)
    new_qubits = new_circuit.all_qubits()

    assert len(new_qubits) == len(circuit.all_qubits()) * num

    expected_qubits = cirq.GridQubit.rect(2 * num, 2)
    expected_circuit = cirq.Circuit(
        cirq.H(expected_qubits[0]),
        cirq.CNOT(expected_qubits[0], expected_qubits[1]),
        cirq.CNOT(expected_qubits[0], expected_qubits[2]),
        cirq.H(expected_qubits[4]),
        cirq.CNOT(expected_qubits[4], expected_qubits[5]),
        cirq.CNOT(expected_qubits[4], expected_qubits[6]),
        cirq.H(expected_qubits[8]),
        cirq.CNOT(expected_qubits[8], expected_qubits[9]),
        cirq.CNOT(expected_qubits[8], expected_qubits[10]),
    )

    new_unitary = cirq.unitary(new_circuit)
    expected_unitary = cirq.unitary(expected_circuit)
    assert np.allclose(new_unitary, expected_unitary)

def test_apply_diagonalizing_gate():
    num_copies = 2
    qubits = cirq.LineQubit.range(2)
    original_circuit = cirq.Circuit(
        cirq.H(qubits[0]),
        cirq.CNOT(qubits[0], qubits[1]),
    )
    N = len(original_circuit.all_qubits())

    circuit = _copy_circuit_parallel(original_circuit, num_copies)

    new_circuit = _apply_diagonalizing_gate(circuit, num_copies)

    # Verify the amount of added operators is N since
    # that is how many diagonalizing gates should be applied
    operations_circuit = sum(1 for _ in circuit.all_operations())
    operations_new_circuit = sum(1 for _ in new_circuit.all_operations())
    assert operations_new_circuit == operations_circuit + N

    # Verify no extra qubits were added
    assert set(new_circuit.all_qubits()) == set(circuit.all_qubits())

    # default diagonalizing matrix for 2 copies
    expected_matrix = np.array(
        [
            [1, 0, 0, 0],
            [0, np.sqrt(2) / 2, np.sqrt(2) / 2, 0],
            [0, np.sqrt(2) / 2, -np.sqrt(2) / 2, 0],
            [0, 0, 0, 1],
        ]
    )

    # fetch the operations in the last moment of the circuit
    last_moment = new_circuit[-1]
    last_moment_ops = list(last_moment.operations)

    # check that the last moment consists of
    # the right amount of diagonalizing gates
    assert len(last_moment_ops) == len(original_circuit.all_qubits())
    for op in last_moment_ops:
        assert op.gate == cirq.MatrixGate(expected_matrix)


def test_generate_diagonalizing_gate():
    expected_matrix = np.array(
        [
            [1, 0, 0, 0],
            [0, np.sqrt(2) / 2, np.sqrt(2) / 2, 0],
            [0, np.sqrt(2) / 2, -np.sqrt(2) / 2, 0],
            [0, 0, 0, 1],
        ]
    )

    assert np.allclose(
        _generate_diagonalizing_gate(2)._matrix, expected_matrix
    )