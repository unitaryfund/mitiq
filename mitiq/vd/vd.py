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

    for i in range(M):
        circuit += rho.transform_qubits(lambda q: qubits[q.x + N*i])

    return circuit

# add unit tests
def test_M_copies_of_rho():
     
    M = 2
    qubits = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(cirq.H(qubits[0]), cirq.CNOT(qubits[0], qubits[1]))
    new_circuit = M_copies_of_rho(circuit, M)

    expected_qubits = cirq.LineQubit.range(2*M)
    expected_circuit = cirq.Circuit(cirq.H(expected_qubits[0]), cirq.CNOT(expected_qubits[0], expected_qubits[1]))
    expected_circuit += cirq.Circuit(cirq.H(expected_qubits[2]), cirq.CNOT(expected_qubits[2], expected_qubits[3]))
    
    assert len(new_circuit) == 4
    assert new_circuit == expected_circuit


    M = 3
    qubits = cirq.LineQubit.range(1)
    circuit = cirq.Circuit(cirq.X(qubits[0]))
    new_circuit = M_copies_of_rho(circuit, M)

    expected_qubits = cirq.LineQubit.range(1*M)
    expected_circuit = cirq.Circuit(cirq.X(expected_qubits[0]))
    expected_circuit += cirq.Circuit(cirq.X(expected_qubits[1]))
    expected_circuit += cirq.Circuit(cirq.X(expected_qubits[2]))
    assert len(new_circuit) == 3
    assert new_circuit == expected_circuit