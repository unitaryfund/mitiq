import cirq
import numpy as np

#
# ======== Many functions are directly taken from VD code ========
#

def M_copies_of_rho(rho: cirq.Circuit, M: int = 2) -> cirq.Circuit:
    """
    Given a circuit rho that acts on N qubits, this function returns a circuit
    that copies rho M times in parallel. This means the resulting circuit has N*M qubits.
    """
    N = len(rho.all_qubits())
    circuit = cirq.Circuit()
    qubits = cirq.LineQubit.range(N * M)

    for i in range(M):
        # Transform each qubit q in rho to q + offset for the i-th copy.
        circuit += rho.transform_qubits(lambda q: qubits[q.x + N*i])
    return circuit


def diagonalize(U: np.ndarray):
    """
    Diagonalize a matrix U (e.g., a 2x2 Pauli or similar) and
    return (V_dagger, eigenvalues) such that U = V diag(eigenvalues) V^dagger.
    """
    eigenvalues, eigenvectors = np.linalg.eigh(U)

    # Sort by ascending phase.
    phases = np.angle(eigenvalues)
    sorted_indices = np.argsort(phases)
    sorted_eigenvalues = eigenvalues[sorted_indices]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]

    # Enforce a phase convention (make first non-zero element positive).
    for i in range(sorted_eigenvectors.shape[1]):
        if np.sign(sorted_eigenvectors[:, i][0]) < 0:
            sorted_eigenvectors[:, i] *= -1

    V_dagger = np.conjugate(sorted_eigenvectors.T)
    
    # Optional consistency check:
    reconstructed = sorted_eigenvectors @ np.diag(sorted_eigenvalues) @ V_dagger
    if not np.allclose(U, reconstructed, atol=1e-7):
        raise ValueError("Diagonalization failed.")
    
    return V_dagger, sorted_eigenvalues


def execute_with_vd(
    input_rho: cirq.Circuit, 
    M: int = 2, 
    K: int = 100, 
    observable: np.ndarray = np.array([[1, 0], [0, -1]])  # Pauli Z by default
):
    """
    Example "VD" function that performs virtual distillation (for M=2 copies),
    measuring the expectation of `observable` on each qubit.
    """
    # input_rho is an N-qubit circuit
    N = len(input_rho.all_qubits())
    rho = M_copies_of_rho(input_rho, M)

    # Example "B gate" for M = 2, as in certain SWAP-based constructions
    Bi_gate = np.array([
        [1, 0,           0,           0],
        [0, np.sqrt(2)/2, np.sqrt(2)/2, 0],
        [0, np.sqrt(2)/2, -np.sqrt(2)/2, 0],
        [0, 0,           0,           1]
    ])

    Ei = [0.0 for _ in range(N)]
    D = 0.0

    # For demonstration, we do K single-shot runs.
    # In practice, you might do more repetitions in a single run, etc.
    for _ in range(K):
        circuit = rho.copy()
        
        # 1) Possibly apply a basis change to measure in the eigenbasis of `observable`.
        #    For Pauli Z, we do nothing. For Pauli X, apply Hadamard, etc.
        #    This is a simplified example using the diagonalization function above.
        if not np.allclose(observable, np.array([[1, 0], [0, -1]])):  # If not Pauli Z
            V_dagger, _ = diagonalize(observable)
            gate = cirq.MatrixGate(V_dagger.conj().T)  # the forward gate is V; we stored V^dagger
            for i in range(M*N):
                circuit.append(gate(cirq.LineQubit(i)))
        
        # 2) Apply the "B gate" entangling each pair (i, i+N) for M=2.
        B_gate = cirq.MatrixGate(Bi_gate)
        for i in range(N):
            circuit.append(B_gate(cirq.LineQubit(i), cirq.LineQubit(i + N)))
        
        # 3) Measure all qubits in Z basis (or as needed for your protocol).
        for i in range(N):
            circuit.append(cirq.measure(cirq.LineQubit(i), key=f"{2*i}"))
        for i in range(N):
            circuit.append(cirq.measure(cirq.LineQubit(i+N), key=f"{2*i+1}"))
        
        # Run the circuit (single shot). 
        simulator = cirq.Simulator()
        result = simulator.run(circuit, repetitions=1)
        
        # Post-processing for a Pauli-Z style measurement:
        #  0 -> +1, 1 -> -1.
        def z_map(bit):
            return 1 if bit == 0 else -1

        z1 = []
        z2 = []
        for i in range(2*N):
            # For each measured qubit i, fetch the single-shot result:
            val = np.squeeze(result.measurements[str(i)])
            if i % 2 == 0:
                z1.append(z_map(val))
            else:
                z2.append(z_map(val))
        
        # Compute numerator contributions E_i (like in the paper)
        for i in range(N):
            productE = 1
            for j in range(N):
                if i != j:
                    productE *= (1 + z1[j] - z2[j] + z1[j]*z2[j])
            Ei[i] += (1 / 2**N) * (z1[i] + z2[i]) * productE
        
        # Compute denominator D
        productD = 1
        for j in range(N):
            productD *= (1 + z1[j] - z2[j] + z1[j]*z2[j])
        D += (1 / 2**N) * productD
    
    # Final corrected expectation values for each qubit
    Z_i_corrected = [Ei[i] / D for i in range(N)]
    return Z_i_corrected


#
# ======== ESD (Exponential Error Suppression) Functions ========
#

def create_derangement_operator(num_qubits: int) -> cirq.Circuit:
    """
    Create a circuit that applies a 'derangement' operator D_n
    as described in ESD protocols. Here, we use a simple chain of SWAPs
    to permute the qubits. Adjust as needed for the actual derangement
    pattern from the paper.
    """
    qubits = [cirq.LineQubit(i) for i in range(num_qubits)]
    derangement_circuit = cirq.Circuit()

    # This is just a simple nearest-neighbor chain of SWAPs
    # (not a full derangement for large n, but a placeholder).
    for i in range(num_qubits - 1):
        derangement_circuit.append(cirq.SWAP(qubits[i], qubits[i + 1]))

    return derangement_circuit


def prepare_n_copies(base_circuit: cirq.Circuit, num_qubits: int, n: int) -> cirq.Circuit:
    """
    Prepare n copies of the quantum state defined by `base_circuit`.
    If base_circuit acts on [0..(num_qubits-1)], then the i-th copy
    acts on [i*num_qubits .. (i+1)*num_qubits-1].
    """
    qubits = [cirq.LineQubit(i) for i in range(num_qubits * n)]
    replicated_circuit = cirq.Circuit()

    for i in range(n):
        offset = i * num_qubits
        replicated_circuit += base_circuit.transform_qubits(
            lambda q: cirq.LineQubit(q.x + offset)
        )

    return replicated_circuit


def measure_expectation_value_with_esd(
    base_circuit: cirq.Circuit,
    observable: np.ndarray,
    num_qubits: int,
    num_copies: int,
    repetitions: int = 1000
) -> float:
    """
    Measure the expectation value of `observable` using an ESD-style approach,
    applying a simple SWAP-based derangement. In practice, the paper's ESD
    protocol often uses a more sophisticated circuit, possibly with an ancilla
    and a controlled-derangement + controlled-observable (Hadamard test).
    
    This simplified function:
      1) Prepares `num_copies` copies of `base_circuit`
      2) Applies a chain of SWAPs as a toy 'derangement'
      3) Measures all qubits in the computational (Z) basis
      4) Maps those 0/1 results to \(\pm 1\) if `observable` ~ Pauli Z
         and returns the average product (as a naive example).
    """
    # 1) Prepare n copies of the circuit
    n_copies_circuit = prepare_n_copies(base_circuit, num_qubits, num_copies)

    # 2) Add the derangement operator
    derangement_circuit = create_derangement_operator(num_copies * num_qubits)

    # Construct full circuit
    full_circuit = cirq.Circuit()
    full_circuit += n_copies_circuit
    full_circuit += derangement_circuit

    # 3) Measure all qubits
    qubits = [cirq.LineQubit(i) for i in range(num_copies * num_qubits)]
    for i, qubit in enumerate(qubits):
        full_circuit.append(cirq.measure(qubit, key=f"m_{i}"))

    # 4) Simulate
    simulator = cirq.Simulator()
    results = simulator.run(full_circuit, repetitions=repetitions)

    # If we are measuring in the computational basis but want a Pauli-Z expectation,
    # map 0 -> +1, 1 -> -1. Then we can average the product. For multi-qubit operators
    # you'll need more careful logic. Below is a trivial single-qubit example.
    # For demonstration, we'll just look at the product across the first `num_qubits`
    # (or some subset). Adjust as needed.
    corrected_sum = 0.0
    for shot_index in range(repetitions):
        # product of Z eigenvalues over the *first* num_qubits (one copy)
        z_product = 1.0
        for i in range(num_qubits):
            bit = results.measurements[f"m_{i}"][shot_index]
            z_val = 1 if bit == 0 else -1
            z_product *= z_val
        corrected_sum += z_product

    # Average
    corrected_expectation = corrected_sum / repetitions
    return corrected_expectation


# # A simple 1-qubit base circuit preparing (|0> + |1>)/sqrt(2).
# base_qubit = cirq.LineQubit(0)
# base_circuit = cirq.Circuit(
#     cirq.H(base_qubit)
# )

# # ESD example usage (toy):
# esd_value = measure_expectation_value_with_esd(
#     base_circuit=base_circuit,
#     observable=np.array([[1, 0], [0, -1]]),  # Pauli Z
#     num_qubits=1,
#     num_copies=2,
#     repetitions=200
# )
# print("ESD estimated expectation (Z):", esd_value)

# # VD example usage (toy):
# vd_value = execute_with_vd(
#     input_rho=base_circuit,
#     M=2,
#     K=50,
#     observable=np.array([[1, 0], [0, -1]])  # Pauli Z
# )
# print("VD corrected expectation (Z) per qubit:", vd_value)
