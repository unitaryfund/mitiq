import cirq
import numpy as np

def M_copies_of_rho(rho: cirq.Circuit, M: int=2):
    '''
    Given a circuit rho that acts on N qubits, this function returns a circuit that copies rho M times in parallel.
    This means the resulting circuit has N * M qubits.
    '''
    
    # if M <= 1:
    #     print("warning: M_copies_of_rho is not needed for M <= 1")
    #     return rho

    N = len(rho.all_qubits())

    circuit = cirq.Circuit()
    qubits = cirq.LineQubit.range(N*M)

    for i in range(M):
        circuit += rho.transform_qubits(lambda q: qubits[q.x + N*i])

    return circuit

