import mitiq
import cirq
import numpy as np

# This virtual distillation works only for M = 2 copies of the state rho
M = 2

Z = np.array([[1, 0], [0, -1]])
X = np.array([[0, 1], [1, 0]])
Y = np.array([[0, -1j], [1j, 0]])
I = np.eye(2)
SWAP = np.array([[1, 0, 0, 0], 
                 [0, 0, 1, 0], 
                 [0, 1, 0, 0], 
                 [0, 0, 0, 1]])

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


# This algorithm only works for M = 2
def generate_swaps(l: list) -> list[tuple]:

    if len(l) % 2 != 0:
        raise ValueError("The list must have an even number of elements, since M=2")

    N = len(l) // 2

    if sorted(l) != list(range(0,2*N)):
        raise ValueError("The list must contain all the integers from 0 to 2*N-1")

    correct_list = []
    for i in range(N):
        correct_list.append(i)
        correct_list.append(i+N)

    swaps = []
    for index, value in enumerate(correct_list):
        if l[index] != value:
            l_index = l.index(value)
            l[index], l[l_index] = l[l_index], l[index]
            swaps.append((index, l_index))


    return swaps

# applies swaps to check if the generate swaps algorithm works
def apply_swaps(swaps_list: list[tuple], list_to_permute: list[int]) -> list[int]:

    permuted_list = list_to_permute.copy()
    for swap in swaps_list:
        permuted_list[swap[0]], permuted_list[swap[1]] = permuted_list[swap[1]], permuted_list[swap[0]]

    return permuted_list


def vd(rho: cirq.Circuit, M: int=2, K: int=100, observable=Z):
    
    # print(f"We run {K} reps which means we need M*K = {M*K} copies of rho")

    # let the circuit be 2 copies of bell state
    N = len(rho.all_qubits())
    rho = M_copies_of_rho(rho, M)

    # Bi corresponding to unitary operator O, which in this case is pauli Z
    Bi_gate = np.array([
            [1, 0, 0, 0],
            [0, np.sqrt(2)/2, np.sqrt(2)/2, 0],
            [0, np.sqrt(2)/2, -np.sqrt(2)/2, 0],
            [0, 0, 0, 1]
        ])


    Ei = [0 for _ in range(N)]
    D = 0
        
    for _ in range(K):
        
        circuit = rho.copy()


        # 1) apply swaps
        swaps = generate_swaps(list(range(2*N)))
        for swap in swaps:
            circuit.append(cirq.SWAP(cirq.LineQubit(swap[0]), cirq.LineQubit(swap[1])))


        # 1.5) apply basis change unitary
        if observable == Z:
            # TODO
            pass


        # 2) apply Bi^(2)
        unitary = Bi_gate
        B_gate = cirq.MatrixGate(unitary)
        for i in range(0,N+1,2):
            circuit.append(B_gate(cirq.LineQubit(i), cirq.LineQubit(i+1)))

        
        # 3) apply measurements
        for i in range(M*N):
            circuit.append(cirq.measure(cirq.LineQubit(i), key=f"{i}"))
        
        
    

        # run the circuit
        simulator = cirq.Simulator()
        result = simulator.run(circuit, repetitions=1)
        
        # print(circuit)

        # post processing measurements
        z1 = []
        z2 = []
        

        for i in range(2*N):
            if i % 2 == 0:
                z1.append(np.squeeze(result.records[str(i)]))
            else:
                z2.append(np.squeeze(result.records[str(i)]))

        # this one is for the pauli Z obvservable
        def map_to_eigenvalues(measurement):
            if measurement == 0:
                return 1
            else:
                return -1
            
        z1 = [map_to_eigenvalues(i) for i in z1]
        z2 = [map_to_eigenvalues(i) for i in z2]
        
        # print(z1)
        # print(z2)

        for i in range(N):
            
            productE = 1
            for j in range(N):
                if i != j:
                    productE *= ( 1 + z1[j] - z2[j] + z1[j]*z2[j] )

            Ei[i] += 1/2**N * (z1[i] + z2[i]) * productE

        productD = 1
        for j in range(N):
            productD *= ( 1 + z1[j] - z2[j] + z1[j]*z2[j] )

        D += 1/2**N * productD 
        
    Z_i_corrected = [Ei[i] / D for i in range(N)]
    # print('Z_i_corrected: ', Z_i_corrected)

    return Z_i_corrected
