import cirq


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

    N = len(circuit.all_qubits())
    new_circuit = cirq.Circuit()
    qubits = cirq.LineQubit.range(N * num_copies)
    for i in range(num_copies):
        new_circuit += circuit.transform_qubits(
            lambda q: qubits[q.x + N * i]
        )  # TODO add compatibility for grid qubits

    return new_circuit
