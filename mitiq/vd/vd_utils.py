from typing import List, cast, Optional
import numpy as np

import cirq


def _copy_circuit_parallel(
    circuit: cirq.Circuit, num_copies: int = 2
) -> cirq.Circuit:
    """Copies a circuit num_copies times in parallel.

    Given a circuit that acts on N qubits,
    this function returns a circuit
    that copies the circuit num_copies times in parallel.
    This means the resulting circuit has N * num_copies qubits.

    Args:
        circuit:
            The circuit to be copied.
        num_copies:
            The number of copies of circuit to be made.

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

def apply_Bi_gate(circuit: cirq.Circuit, gate: Optional[np.ndarray] = None) -> cirq.Circuit:
    """
    Apply a Bi gate to a circuit. We assume that N is even (from the other functions).

    Args:
        circuit:
            The circuit to apply the Bi gate to.
        gate:
            The matrix representation of the Bi gate.
            Default is None, in which case the pre-defined Bi gate will be applied.

    Returns:
        The circuit with the Bi gate applied.
    """
    if gate is None:
        gate = np.array([
            [1, 0, 0, 0],
            [0, np.sqrt(2)/2, np.sqrt(2)/2, 0],
            [0, np.sqrt(2)/2, -np.sqrt(2)/2, 0],
            [0, 0, 0, 1]
        ])

    B_gate = cirq.MatrixGate(gate)

    N = len(circuit.all_qubits())

    for i in N:
        circuit.append(B_gate(cirq.LineQubit(i), cirq.LineQubit(i+N)))

    return circuit