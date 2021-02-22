import cirq
from mitiq.mitiq_qiskit.conversions import to_qiskit, from_qiskit
from mitiq._typing import QPROGRAM
from qiskit import QuantumCircuit, compiler
from clifford_training_data import (count_non_cliffords,
                                    generate_training_circuits)
import numpy as np
import matplotlib.pyplot as plt


def random_circuit(
    qubits: int,
    depth: int,
) -> QPROGRAM:
    '''Function to generate a random quantum circuit in cirq. The circuit is \
       based on the hardware efficient ansatz,
    with alternating CNOT layers with randomly selected single qubit gates in \
    between.
    Args:
        qubits: number of qubits in circuit.
        depth: depth of the RQC.
    Returns:
        cirquit: a random quantum circuit of specified depth.
    '''
    # Get a rectangular grid of qubits.
    qubits = cirq.GridQubit.rect(qubits, 1)
    # Generates a random circuit on the provided qubits.
    circuit = \
        cirq.experiments.\
        random_rotations_between_grid_interaction_layers_circuit(
            qubits=qubits, depth=depth, seed=0)
    circuit.append(cirq.measure(*qubits, key='z'))
    return(circuit)


def qiskit_circuit_transplation(
    circ: QPROGRAM,
) -> QPROGRAM:
    """Decomposes qiskit circuit object into Rz, Rx(pi/2) (sx), X and CNOT \
       gates.
    Args:
        circ: original circuit of interest assumed to be qiskit circuit object.
    Returns:
        circ_new: new circuite compiled and decomposed into the above gate set.
    """
    circ = compiler.transpile(circ, basis_gates=['u3', 'cx'],
                              optimization_level=3)
    circ_new = QuantumCircuit(len(circ.qubits), len(circ.clbits))
    for i in range(len(circ.data)):
        # get information for the gate
        gate = circ.data[i][0]
        name = gate.name
        if name == 'cx':
            qubit = [circ.data[i][1][0].index, circ.data[i][1][1].index]
            circ_new.cx(qubit[0], qubit[1])
        elif name == 'u3':
            qubit = circ.data[i][1][0].index
            parameters = gate.params
            # doing the decompostion of u3 gate here:
            theta = parameters[0]
            phi = parameters[1]
            lamda = parameters[2]
            circ_new.rz(phi+np.pi, qubit)
            circ_new.rx(np.pi/2, qubit)
            circ_new.rz(theta+np.pi, qubit)
            circ_new.rx(np.pi/2, qubit)
            circ_new.rz(lamda, qubit)
        elif name == 'measure':
            qubit = circ.data[i][1][0].index
            cbit = circ.data[i][2][0].index
            circ_new.measure(qubit, cbit)
    return(circ_new)


def uniform_circuit(
) -> QPROGRAM:
    """Returns a single qubit circuit with a uniform distribution of theta
    rotations about the z axis.
    Returns:
        circ: qiskit quantum circuit object.
    """
    circ = QuantumCircuit(1, 1)
    for i in range(2001):
        angle = (i*np.pi*4)/(2000)
        circ.rz(angle, 0)
    circ.measure(0, 0)
    return(circ)


num_training_circuits = 10
fraction_non_clifford = 0.90
# num_qubits = 8
# layers = 8
# circuit = cirq.circuits.Circuit(random_circuit(num_qubits, layers))
# circuit = from_qiskit(qiskit_circuit_transplation(to_qiskit(circuit)))
circuit = from_qiskit(uniform_circuit())
non_cliffords = count_non_cliffords(circuit)
print('number of non Clifford gates ORIG = ', non_cliffords)
additional_options = {'sigma_replace': 0.5, 'sigma_select': 0.5}
# print(additional_options)
(training_circuits, angles_original_list,
 angles_replaced_list) = generate_training_circuits(
                                        circuit,
                                        num_training_circuits,
                                        fraction_non_clifford,
                                        method_select='probabilistic',
                                        method_replace='probabilistic',
                                        additional_options=additional_options)

# testing circuits look good. Note that a figure is saved with every circuit
# in the trianing set for visual inspection
angles_selected = np.array(angles_original_list).flatten()
angles_replaced = np.array(angles_replaced_list).flatten()


def plot_histogram_of_angles():
    plt.figure(1)
    plt.hist(abs(angles_selected), bins=100)
    plt.ylabel('counts')
    plt.xlabel('angle')
    plt.show()


def plot_circuits():
    for i, training_circuit in enumerate(training_circuits):
        non_cliffords = count_non_cliffords(training_circuit)
        print('number of non Clifford gates = ', non_cliffords)
        training_circuit = to_qiskit(training_circuit)
        training_circuit.draw(output='mpl', filename='my_circuit_%s.png' % (i))

    to_qiskit(circuit).draw(output='mpl', filename='my_circuit_orig.png')


plot_histogram_of_angles()
