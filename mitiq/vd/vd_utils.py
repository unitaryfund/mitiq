import cirq
import numpy as np

def M_copies_of_rho(rho: cirq.Circuit, M: int=2) -> cirq.Circuit:
    """ Copies a circuit M times in parallel.

    Given a circuit rho that acts on N qubits, this function returns a circuit that copies rho M times in parallel.
    This means the resulting circuit has N * M qubits.

    Args:
        rho: 
            The circuit to be copied.
        M:
            The number of copies of rho to be made.

    Returns:
        A cirq circuit that is the parallel composition of M copies of rho.
    """
    
    # if M <= 1:
    #     print("warning: M_copies_of_rho is not needed for M <= 1")
    #     return rho

    N = len(rho.all_qubits())

    circuit = cirq.Circuit()
    qubits = cirq.LineQubit.range(N*M)

    unitary_gate = cirq.MatrixGate(cirq.unitary(rho))
    for i in range(M):
        qubit_subset = qubits[i * N : (i+1) * N]
        circuit.append(unitary_gate(*qubit_subset))
        
    return circuit