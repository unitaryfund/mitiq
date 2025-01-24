import cirq
from typing import List, Union


def _copy_circuit_parallel(circuit: cirq.Circuit,
                           num_copies: int = 2) -> cirq.Circuit:
    """Copies a circuit M times in parallel.

    Given a circuit rho that acts on N qubits,
    this function returns a circuit that copies rho M times in parallel.
    This means the resulting circuit has N * M qubits.

    Args:
        rho:
            The circuit to be copied.
        M:
            The number of copies of rho to be made.

    Returns:
        A cirq circuit that is the parallel composition of M copies of rho.
    """

    new_circuit = cirq.Circuit()
    N = len(circuit.all_qubits())
    qubits: List[Union[
        cirq.LineQubit, cirq.GridQubit]] = list(circuit.all_qubits())

    # LineQubits
    if isinstance(qubits[0], cirq.LineQubit):

        new_qubits = cirq.LineQubit.range(N * num_copies)
        for i in range(num_copies):
            new_circuit += circuit.transform_qubits(
                lambda q: new_qubits[qubits.index(q) + i * N]
            )

    # GridQubits
    elif isinstance(qubits[0], cirq.GridQubit):

        grid_rows = max([q.row + 1 for q in qubits])
        grid_cols = max([q.col + 1 for q in qubits])

        new_qubits = cirq.GridQubit.rect(grid_rows * num_copies, grid_cols)
        for i in range(num_copies):
            new_circuit += circuit.transform_qubits(
                lambda q: cirq.GridQubit(q.row + i * grid_rows, q.col)
            )

    return new_circuit
