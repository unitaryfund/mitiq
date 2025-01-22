import mitiq
import cirq
import numpy as np

# This virtual distillation works only for M = 2 copies of the state rho
M = 2

Z = np.array([[1, 0], [0, -1]])
X = np.array([[0, 1], [1, 0]])

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

def diagonalize(U: np.ndarray) -> np.ndarray:
    """Diagonalize a density matrix rho and return the basis change unitary V†."""
    
    eigenvalues, eigenvectors = np.linalg.eigh(U)
    
    # Sort eigenvalues and eigenvectors by ascending phase
    phases = np.angle(eigenvalues)
    sorted_indices = np.argsort(phases)
    sorted_eigenvalues = eigenvalues[sorted_indices]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]
    
    # Normalize and enforce sign convention (optional)
    for i in range(sorted_eigenvectors.shape[1]):
        # Force the first nonzero element of each eigenvector to be positive
        if np.sign(sorted_eigenvectors[:, i][0]) < 0:
            sorted_eigenvectors[:, i] *= -1
    
    # Compute V† (conjugate transpose of V)
    V_dagger = np.conjugate(sorted_eigenvectors.T)
    
    # check
    if not np.allclose(U, np.dot(sorted_eigenvectors, np.dot(np.diag(sorted_eigenvalues), V_dagger))):
        raise ValueError("Diagonalization failed.")

    return V_dagger, sorted_eigenvalues

def execute_with_vd(input_rho: cirq.Circuit, M: int=2, K: int=100, observable=Z):
    '''
    Given a circuit rho that acts on N qubits, this function returns the expectation values of a given observable for each qubit i. 
    The expectation values are corrected using the virtual distillation algorithm. 
    '''

    # input rho is an N qubit circuit
    N = len(input_rho.all_qubits())
    rho = M_copies_of_rho(input_rho, M)

    # Coupling unitary corresponding to the diagonalization of the SWAP (as seen in the paper) for M = 2:
    Bi_gate = np.array([
            [1, 0, 0, 0],
            [0, np.sqrt(2)/2, np.sqrt(2)/2, 0],
            [0, np.sqrt(2)/2, -np.sqrt(2)/2, 0],
            [0, 0, 0, 1]
        ])

    Ei = [0 for _ in range(N)]
    D = 0
    
    # Forcing odd K, this is a workaround so that D (see end of the function) cannot be 0 accidentally
    if K%2 == 0:
        K += 1

    # Helper function to map the results to the eigenvalues of the pauli Z observable
    def map_to_eigenvalues(measurement):
        if measurement == 0:
            return 1
        else:
            return -1
        

    
    # prepare gates
    B_gate = cirq.MatrixGate(Bi_gate)

    basis_change_unitary = diagonalize(observable)[0]
    if not np.allclose(observable, Z):
        gate = cirq.MatrixGate(basis_change_unitary)
    else: 
        gate = cirq.I()
    
    for _ in range(K):
        
        circuit = rho.copy()

        for i in range(N):
            # 1) apply basis change unitary to all M * N qubits
            for m in range(M):
                circuit.append(gate(cirq.LineQubit(i + m*N)))

            # 2) apply the diagonalization gate B
            # [implementation works only for the M=2 case]
            circuit.append(B_gate(cirq.LineQubit(i), cirq.LineQubit(i+N)))

        
            # 3) apply measurements
            # the measurement keys are applied in accordance with the SWAPS that are applied in the pseudo code in the paper.
            # The SWAP operations are omitted here since they are hardware specific.
            # [once again this specific code is for M = 2]
            circuit.append(cirq.measure(cirq.LineQubit(i), key=f"{2*i}"))
            circuit.append(cirq.measure(cirq.LineQubit(i+N), key=f"{2*i+1}"))
        
        # run the circuit
        simulator = cirq.Simulator()
        result = simulator.run(circuit, repetitions=1)
                
        # post processing measurements
        z1 = []
        z2 = []
        
        for i in range(2*N):
            if i % 2 == 0:
                z1.append(np.squeeze(result.records[str(i)]))
            else:
                z2.append(np.squeeze(result.records[str(i)]))
    
        z1 = [map_to_eigenvalues(i) for i in z1]
        z2 = [map_to_eigenvalues(i) for i in z2]

        # Part obtained from the pseudocode of the paper
        productD = 1
        for i in range(N):
            productE = 1
            for j in range(N):
                if i != j:
                    productE *= ( 1 + z1[j] - z2[j] + z1[j]*z2[j] )
            Ei[i] += 1/2**N * (z1[i] + z2[i]) * productE

            productD *= ( 1 + z1[j] - z2[j] + z1[j]*z2[j] )
        
        D += 1/2**N * productD 
    
    # K must be odd so that D cannot accidentally be 0 and give an error.
    # [Forcing odd K is a workaround, we should look into this]
    Z_i_corrected = [Ei[i] / D for i in range(N)]

    return Z_i_corrected
