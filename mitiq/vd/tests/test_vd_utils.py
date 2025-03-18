import cirq
import numpy as np

from mitiq.vd.vd_utils import (
    _apply_cyclic_system_permutation,
    _apply_symmetric_observable,
    _copy_circuit_parallel,
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


def test_apply_cyclic_system_permutation():
    """Test with N_qubits=1, M=2 (4x4 matrix)"""
    # Create an 4x4 matrix with values 1 to 17
    matrix = np.arange(1, 17).reshape(4, 4)

    # Expected result
    expected = matrix[[0, 2, 1, 3]]

    result = _apply_cyclic_system_permutation(matrix, N_qubits=1)
    assert np.allclose(result, expected)


def test_edge_case_single_qubit():
    """Test with N_qubits=1, M=1 (minimal case)"""
    matrix = np.array([[1, 2], [3, 4]])
    # No permutation should happen here
    expected = matrix.copy()
    result = _apply_cyclic_system_permutation(matrix, N_qubits=1, M=1)
    assert np.allclose(result, expected)


def test_apply_symmetric_observable():
    """Test applying Z observable to 2D matrix"""
    # For N_qubits=1 and M=2, we're dealing with 4x4 matrices
    matrix = np.arange(1, 17).reshape(4, 4)

    # Z observable is [[1, 0], [0, -1]]
    # For N_qubits=1, we expect one output matrix with diagonals scaled
    # by +1 for |0⟩ state and -1 for |1⟩ state
    expected = np.array(
        [
            [
                [1.0, 0.0, 0.0, -4.0],
                [5.0, 0.0, 0.0, -8.0],
                [9.0, 0.0, 0.0, -12.0],
                [13.0, 0.0, 0.0, -16.0],
            ]
        ]
    )

    result = _apply_symmetric_observable(matrix, N_qubits=1)
    assert np.allclose(result, expected)


def test_apply_symmetric_observable_X():
    """Test applying X observable to 2D matrix"""
    # For N_qubits=1 and M=2, we're dealing with 4x4 matrices
    matrix = np.arange(1, 17).reshape(4, 4)
    observable = np.array([[0, 1], [1, 0]])

    # For N_qubits=1, we expect one output matrix with diagonals scaled
    # by +1 for |0⟩ state and -1 for |1⟩ state
    expected = np.array(
        [
            [
                [7.0, 8.0, 9.0, 10.0],
                [7.0, 8.0, 9.0, 10.0],
                [7.0, 8.0, 9.0, 10.0],
                [7.0, 8.0, 9.0, 10.0],
            ]
        ]
    )

    result = _apply_symmetric_observable(
        matrix, N_qubits=1, observable=observable
    )
    assert np.allclose(result, expected)
