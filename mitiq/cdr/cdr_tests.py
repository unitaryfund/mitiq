import cirq
from mitiq.mitiq_qiskit.conversions import to_qiskit, from_qiskit
from mitiq._typing import QPROGRAM
from qiskit import QuantumCircuit, compiler
from clifford_training_data import count_non_cliffords, \
    generate_training_circuits
import numpy as np


def random_circuit(
    qubits: int,
    depth: int,
    measure: bool,
) -> QPROGRAM:
    '''Function to generate a random quantum circuit in cirq. The circuit is \
       based on the hardware efficient ansatz,
    with alternating CNOT layers with randomly selected single qubit gates in \
    between.
    Args:
        qubits: number of qubits in circuit.
        depth: depth of the RQC.
        measure: measurements or not.
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
    if measure:
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
    # this decomposes the circuit into u3 and cnot gates:
    circ = compiler.transpile(
        circ, basis_gates=['sx', 'rz', 'cx', 'x'], optimization_level=3)
    # print(circ.draw())
    # now for each U3(theta, phi, lambda), this can be converted into
    # Rz(phi+pi)Rx(pi/2)Rz(theta+pi)Rx(pi/2)Rz(lambda)
    try:
        circ_new = QuantumCircuit(len(circ.qubits), len(circ.clbits))
    except:
        circ_new = QuantumCircuit(len(circ.qubits))
    for i in range(len(circ.data)):
        # get information for the gate
        gate = circ.data[i][0]
        name = gate.name
        if name == 'cx':
            qubit = [circ.data[i][1][0].index, circ.data[i][1][1].index]
            parameters = []
            circ_new.cx(qubit[0], qubit[1])
        if name == 'rz':
            parameters = (float(gate.params[0])) % (2 * np.pi)
            # leave out empty Rz gates:
            if parameters != 0:
                qubit = circ.data[i][1][0].index
                circ_new.rz(parameters, qubit)
        if name == 'sx':
            parameters = np.pi/2
            qubit = circ.data[i][1][0].index
            circ_new.rx(parameters, qubit)
        if name == 'x':
            qubit = circ.data[i][1][0].index
            circ_new.x(qubit)
        elif name == 'measure':
            qubit = circ.data[i][1][0].index
            cbit = circ.data[i][2][0].index
            circ_new.measure(qubit, cbit)
    return(circ_new)


num_qubits = 4
layers = 1
measure = True
num_training_circuits = 2
fraction_non_clifford = 0.5
circuit = cirq.circuits.Circuit(random_circuit(num_qubits, layers, measure))
circuit = from_qiskit(qiskit_circuit_transplation(to_qiskit(circuit)))
non_cliffords = count_non_cliffords(circuit)
print('number of non Clifford gates ORIG = ', non_cliffords)
# additional_options = {'sigma_replace': 0.1 }
# print(additional_options)
training_circuits = generate_training_circuits(
    circuit, num_training_circuits, fraction_non_clifford,
    method_select='random', method_replace='probabilistic')

# testing circuits look good. Note that a figure is saved with every circuit
# in the trianing set for visual inspection
for i, training_circuit in enumerate(training_circuits):
    non_cliffords = count_non_cliffords(training_circuit)
    print('number of non Clifford gates = ', non_cliffords)
    training_circuit = to_qiskit(training_circuit)
    training_circuit.draw(output='mpl', filename='my_circuit_%s.png' % (i))

to_qiskit(circuit).draw(output='mpl', filename='my_circuit_orig.png')
