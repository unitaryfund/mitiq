import numpy as np
from pyquil import get_qc, Program
from pyquil.gates import RX, RY, I, H, X, CNOT, MEASURE
from typing import List, Union
from collections import Counter
import mitiq
from mitiq import zne
from scipy import optimize
from pyquil.paulis import PauliTerm, PauliSum, sZ
from pyquil.noise import pauli_kraus_map, append_kraus_to_gate

# initialize the quantum device
qc = get_qc("2q-qvm")


program = Program()
theta = program.declare("theta", memory_type="REAL")
program += RX(theta, 0)

# # Simulate depolarizing noise with custom gate
# # this is to check if adding custom gate works outside of executor function

# # Mimic "define_noisy_gate" example, but with depolarizing noise
# prob = 0.1
# num_qubits = 1
# d = 4**num_qubits
# kraus_list = [(1-prob)/d]*d
# kraus_list[0] += prob
# kraus_ops = pauli_kraus_map(kraus_list)
# program.define_noisy_gate("X", [0], append_kraus_to_gate(kraus_ops, np.array([[0, 1],
#                                                 [1, 0]])))    

# # Usage as indicated in "define_noisy_gate" docstring
# program.define_noisy_gate("X", [0], kraus_ops)
  

hamiltonian = sZ(0)
samples = 2000
pauli_sum = PauliSum([hamiltonian])
for j, term in enumerate(pauli_sum.terms):
    meas_basis_change = Program()
    marked_qubits = []
    for index, gate in term:
        marked_qubits.append(index)
        if gate == 'X':
            meas_basis_change.inst(RY(-np.pi / 2, index))
        elif gate == 'Y':
            meas_basis_change.inst(RX(np.pi / 2, index))
        program += meas_basis_change
    # Memory declaration for readout "ro"
    ro = program.declare('ro', 'BIT', max(marked_qubits) + 1)
program.wrap_in_numshots_loop(samples)
quil_prog = qc.compiler.quil_to_native_quil(program, protoquil=True)
    

def executor(thetas, qc, ro, samples: int,
                pauli_sum: Union[PauliSum, PauliTerm, np.ndarray],
                pyquil_prog: Program) -> float:
    """
    Compute the expectation value of pauli_sum over the distribution generated from pyquil_prog.

    :param pyquil_prog: The state preparation Program to calculate the expectation value of.
    :param pauli_sum: PauliSum representing the operator of which to calculate the expectation
            value
    :param samples: The number of samples used to calculate the expectation value.
    :param qc: The QuantumComputer object.
    :return: A float representing the expectation value of pauli_sum given the distribution
            generated from quil_prog.
        """
    # # Simulate depolarizing noise. 
    # # Requires use of custom gates which cannot be folded and must be defined in executor fn 
    noisy = pyquil_prog.copy()  
    prob = 0.1
    num_qubits = 1
    d = 4**num_qubits
    kraus_list = [(1-prob)/d]*d
    kraus_list[0] += prob
    kraus_ops = pauli_kraus_map(kraus_list)
    noisy.define_noisy_gate("X", [0], append_kraus_to_gate(kraus_ops, np.array([[0, 1],
                                                [1, 0]])))                                         
    # # Ideal X gate for comparison
    # noisy += X(0)
    noisy += [MEASURE(qubit, r)
         for qubit, r in zip(list(range(max(marked_qubits) + 1)), ro)]
    expectation = 0.0
    pauli_sum = PauliSum([pauli_sum])
    for j, term in enumerate(pauli_sum.terms):
        qubits_to_measure = []
        for index, gate in term:
            qubits_to_measure.append(index)    
            meas_outcome = expectation_from_sampling(
                            thetas, noisy, qubits_to_measure, 
                            qc, samples)
            expectation += term.coefficient * meas_outcome
    return expectation.real

def expectation_from_sampling(thetas, executable: Program,
                              marked_qubits: List[int], qc,
                              samples: int) -> float:
    """

    Given a wavefunctions, this calculates the expectation value of the Zi
    operator where i ranges over all the qubits given in marked_qubits.

    :param pyquil_program: pyQuil program generating some state
    :param marked_qubits: The qubits within the support of the Z pauli
                          operator whose expectation value is being calculated
    :param qc: A QuantumComputer object.
    :param samples: Number of bitstrings collected to calculate expectation
                    from sampling.
    :returns: The expectation value as a float.
    """
    bitstring_samples = qc.run(executable, memory_map={'theta': thetas})
    bitstring_tuples = list(map(tuple, bitstring_samples))

    freq = Counter(bitstring_tuples)

    # perform weighted average
    exp_val = 0
    for bitstring, count in freq.items():
        bitstring_int = int("".join([str(x) for x in bitstring[::-1]]), 2)
        if parity_even_p(bitstring_int, marked_qubits):
            exp_val += float(count) / samples
        else:
            exp_val -= float(count) / samples
    return exp_val


def parity_even_p(state, marked_qubits):
    """
    Parity is relative to the binary representation of the integer state.

    :param state: The wavefunction index that corresponds to this state.
    :param marked_qubits: The indexes to be considered in the parity sum.
    :returns: A boolean corresponding to the parity.
    """
    mask = 0
    for q in marked_qubits:
        mask |= 1 << q
    return bin(mask & state).count("1") % 2 == 0



thetas = np.linspace(0, 2 * np.pi, 101)
results = []
 
for theta in thetas:
    results.append(executor(theta, qc, ro, samples, hamiltonian, quil_prog))

from matplotlib import pyplot as plt
_ = plt.figure(1)
_ = plt.plot(thetas, results, 'o-')
_ = plt.xlabel(r'$\theta$', fontsize=18)
_ = plt.ylabel(r'$\langle \Psi(\theta) | Z | \Psi(\theta) \rangle$', fontsize=18)    
_ = plt.title('Noisy Energy Landscape')
plt.show()




