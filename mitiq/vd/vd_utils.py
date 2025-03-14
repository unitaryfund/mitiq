from typing import List, cast

import cirq
import numpy as np


def _copy_circuit_parallel(
    circuit: cirq.Circuit, num_copies: int = 2
) -> cirq.Circuit:
    """Given a circuit that acts on N qubits, this function returns a circuit
    that copies the circuit num_copies times in parallel. This means the
    resulting circuit has N * num_copies qubits.

    Args:
        circuit: The circuit to be copied.
        num_copies: The number of copies of circuit to be made.

    Returns:
        A cirq circuit that is the parallel composition of
          num_copies copies of circuit.
    """

    new_circuit = cirq.Circuit()
    N = len(circuit.all_qubits())
    qubits = list(circuit.all_qubits())

    # LineQubits
    if isinstance(qubits[0], cirq.LineQubit):

        def map_for_line_qubits(q: cirq.Qid) -> cirq.Qid:
            assert isinstance(q, cirq.LineQubit)
            return cirq.LineQubit(q.x + N * i)

        for i in range(num_copies):
            new_circuit += circuit.transform_qubits(map_for_line_qubits)

    # GridQubits
    elif isinstance(qubits[0], cirq.GridQubit):
        qubits_cast_grid = cast(List[cirq.GridQubit], qubits)
        grid_rows = max([qu.row + 1 for qu in qubits_cast_grid])

        def map_for_grid_qubits(qu: cirq.Qid) -> cirq.Qid:
            assert isinstance(qu, cirq.GridQubit)
            return cirq.GridQubit(qu.row + grid_rows * i, qu.col)

        for i in range(num_copies):
            new_circuit += circuit.transform_qubits(map_for_grid_qubits)

    return new_circuit


def _apply_diagonalizing_gate(
    circuit: cirq.Circuit, num_copies: int
) -> cirq.Circuit:
    """Apply the diagonalizing gate to a circuit, assuming the circuit has been
    copied in parallel ``num_copies`` times.

    The diagonalizing gate is applied to the first qubit of all copies, to the
    second qubit of all copies, etc.

    Args:
        circuit: The circuit to apply the diagonalizing gate to.
        num_copies: The number of copies of the original circuit that the input
            circuit consists of.

    Returns:
        The circuit with the diagonalizing gate applied.
    """

    new_circuit = circuit.copy()

    # this is the number of qubits in the original circuit
    N = len(circuit.all_qubits()) // num_copies

    diag_gate = _generate_diagonalizing_gate(num_copies)

    for i in range(N):
        qubits = [
            cirq.LineQubit(i + N * j) for j in range(num_copies)
        ]  # select qubit i of each copy

        new_circuit.append(diag_gate(*qubits))

    return new_circuit


def _generate_diagonalizing_gate(num_copies: int = 2) -> cirq.Gate:
    """Generate the diagonalizing gate for the VD algorithm. Currently only
    ``num_copies=2`` is supported.

    Args:
        num_qubits: The number of qubits that the gate acts on.

    Returns: The diagonalizing gate.
    """
    if num_copies == 2:
        diagonalizing_matrix = np.array(
            [
                [1, 0, 0, 0],
                [0, np.sqrt(2) / 2, np.sqrt(2) / 2, 0],
                [0, np.sqrt(2) / 2, -np.sqrt(2) / 2, 0],
                [0, 0, 0, 1],
            ]
        )
    else:
        raise NotImplementedError(
            "Only num_copies = 2 is currently supported."
        )

    return cirq.MatrixGate(diagonalizing_matrix)
