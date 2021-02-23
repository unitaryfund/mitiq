import cirq
from random import randint, uniform
import numpy as np
from mitiq._typing import QPROGRAM
from clifford_training_data import (_array_to_circuit, _circuit_to_array,
                                    _is_clifford_angle,
                                    _map_to_near_clifford,
                                    _closest_clifford,
                                    _random_clifford,
                                    _angle_to_probabilities,
                                    _probabilistic_angle_to_clifford,
                                    count_non_cliffords,
                                    generate_training_circuits)
from qiskit import compiler, QuantumCircuit
from mitiq.mitiq_qiskit.conversions import to_qiskit, from_qiskit


"""Tests for training circuits generation for Clifford data regression.
"""
CLIFFORD_ANGLES = (0.0, np.pi/2, np.pi, (3/2)*(np.pi))


def random_circuit(
    qubits: int,
    depth: int,
) -> QPROGRAM:
    """Function to generate a random quantum circuit in cirq. The circuit is
       based on the hardware efficient ansatz,
    with alternating CNOT layers with randomly selected single qubit gates in
    between.
    Args:
        qubits: number of qubits in circuit.
        depth: depth of the RQC.
    Returns:
        cirquit: a random quantum circuit of specified depth.
    """
    # Get a rectangular grid of qubits.
    qubits = cirq.GridQubit.rect(qubits, 1)
    # Generates a random circuit on the provided qubits.
    circuit = cirq.experiments.\
        random_rotations_between_grid_interaction_layers_circuit(
            qubits=qubits, depth=depth, seed=0)
    circuit.append(cirq.measure(*qubits, key='z'))
    return(circuit)


def qiskit_circuit_transpilation(
    circ: QPROGRAM,
) -> QuantumCircuit:
    """Decomposes qiskit circuit object into Rz, Rx(pi/2) (sx), X and CNOT
       gates.
    Args:
        circ: original circuit of interest assumed to be qiskit circuit object.
    Returns:
        circ_new: new circuite compiled and decomposed into the above gate set.
    """
    circ = compiler.transpile(circ, basis_gates=['u3', 'cx'],
                              optimization_level=3)
    circ_new = QuantumCircuit(len(circ.qubits), len(circ.clbits))
    for (gate, qubits, cbits) in circ.data:
        # get information for the gate
        name = gate.name
        if name == 'cx':
            qubit = [qubits[0].index, qubits[1].index]
            circ_new.cx(qubit[0], qubit[1])
        elif name == 'u3':
            qubit = qubits[0].index
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
            qubit = qubits[0].index
            cbit = cbits[0].index
            circ_new.measure(qubit, cbit)
    return(circ_new)


num_qubits = 4
layers = 10
num_training_circuits = 10
fraction_non_clifford = 0.3
circuit = cirq.circuits.Circuit(
    random_circuit(num_qubits, layers))
circuit = from_qiskit(qiskit_circuit_transpilation(to_qiskit(circuit)))
non_cliffords = count_non_cliffords(circuit)


def test_generate_training_circuits():
    """Test that generate_training_circuits function is working properly with
       the random projrection method.
    """
    method_select_options_list = ['random', 'probabilistic']
    method_replace_options_list = ['random', 'probabilistic', 'closest']
    additional_options = {'sigma_select': 0.5, 'sigma_replace': 0.5}
    non_cliffords = count_non_cliffords(circuit)
    for method_select in method_select_options_list:
        for method_replace in method_replace_options_list:
            test_training_set_circuits = generate_training_circuits(
                circuit, num_training_circuits, fraction_non_clifford,
                method_select, method_replace)[0]
            test_training_set_circuits_with_options = (
                generate_training_circuits(
                    circuit, num_training_circuits,
                    fraction_non_clifford, method_select,
                    method_replace, additional_options=additional_options))[0]
            assert len(test_training_set_circuits) == num_training_circuits

            assert (len(
                test_training_set_circuits_with_options)
                == num_training_circuits)

            for i in range(num_training_circuits):
                assert (count_non_cliffords(test_training_set_circuits[i]) ==
                        int(fraction_non_clifford*non_cliffords))
                assert len(test_training_set_circuits[i]) == len(circuit)
                assert len(test_training_set_circuits[i].all_qubits()) == len(
                    circuit.all_qubits())
                assert (count_non_cliffords(
                    test_training_set_circuits_with_options[i]) ==
                    int(fraction_non_clifford*non_cliffords))
                assert len(test_training_set_circuits_with_options[i]) == len(
                    circuit)
                assert (len(
                    test_training_set_circuits_with_options[i].all_qubits())
                    == len(circuit.all_qubits()))


def test_map_to_near_cliffords():
    method_select_options_list = ['random', 'probabilistic']
    method_replace_options_list = ['random', 'probabilistic', 'closest']
    additional_options = {'sigma_select': 0.5, 'sigma_replace': 0.5}
    data, empty_circuit = _circuit_to_array(circuit)
    mask_rz = data[1, :] == 'rz'
    rz_circ_data = data[:, mask_rz]
    mask_not_rz = data[1, :] != 'rz'
    not_rz_circ_data = data[:, mask_not_rz]
    mask_non_cliff = _is_clifford_angle(rz_circ_data[2, :])
    mask_non_cliff = ~mask_non_cliff
    rz_non_cliff = rz_circ_data[:, mask_non_cliff]
    mask_cliff = _is_clifford_angle(rz_circ_data[2, :])
    rz_cliff = rz_circ_data[:, mask_cliff]
    total_non_cliff = len(rz_non_cliff[0])
    # find all the non-Clifford gates:
    all_cliff = np.column_stack((not_rz_circ_data, rz_cliff))
    non_cliffords = count_non_cliffords(circuit)
    for method_select in method_select_options_list:
        for method_replace in method_replace_options_list:
            projected_circuit = _map_to_near_clifford(
                rz_non_cliff, all_cliff, empty_circuit, total_non_cliff,
                fraction_non_clifford, method_select, method_replace)[0]

            projected_circuit_with_options = _map_to_near_clifford(
                rz_non_cliff, all_cliff, empty_circuit, total_non_cliff,
                fraction_non_clifford, method_select, method_replace,
                additional_options=additional_options)[0]
            assert count_non_cliffords(projected_circuit) == int(
                fraction_non_clifford*non_cliffords)
            assert len(projected_circuit) == len(circuit)
            assert len(projected_circuit.all_qubits()
                       ) == len(circuit.all_qubits())
            assert count_non_cliffords(projected_circuit_with_options) == int(
                fraction_non_clifford*non_cliffords)
            assert len(projected_circuit_with_options) == len(circuit)
            assert len(projected_circuit_with_options.all_qubits()
                       ) == len(circuit.all_qubits())


def test_count_non_cliffords():
    number_non_cliffords = 0
    example_circuit = QuantumCircuit(1)
    for i in range(100):
        rand = randint(1, 2)
        rand2 = randint(1, 4)-1
        if rand % 2 == 0:
            example_circuit.rz(CLIFFORD_ANGLES[rand2], 0)
        else:
            example_circuit.rz(uniform(0, 2*np.pi), 0)
            number_non_cliffords += 1
        example_circuit = from_qiskit(example_circuit)
        assert count_non_cliffords(example_circuit) == number_non_cliffords
        example_circuit = to_qiskit(example_circuit)


def test_circuit_to_array():
    array = _circuit_to_array(circuit)
    empty_circuit = array[1]
    array = array[0]
    assert len(array[0, :]) == len(array[1, :])
    assert len(array[0, :]) == len(array[2, :])
    assert len(array[0, :]) == len(array[3, :])
    assert len(array[0, :]) == len(array[4, :])
    assert isinstance(empty_circuit, cirq.circuits.circuit.Circuit)


def test_array_to_circuit():
    data, empty_circuit = _circuit_to_array(circuit)
    circuit_2 = _array_to_circuit(data, empty_circuit)
    for s in range(len(circuit_2)):
        assert circuit_2[s] == circuit[s]


def test_is_clifford_angle():
    cliff_angs = np.array(CLIFFORD_ANGLES)

    for i in range(15):
        assert _is_clifford_angle(int(i)*cliff_angs).all()
        ang = uniform(0, 2*np.pi)
        assert not _is_clifford_angle(ang)


def test_closest_clifford():
    for ang in CLIFFORD_ANGLES:
        angs = np.linspace(ang - np.pi/4+0.01, ang+np.pi/4-0.01)
        for a in angs:
            assert _closest_clifford(a) == ang


def test_random_clifford():
    for ang in CLIFFORD_ANGLES:
        assert _random_clifford(ang) in np.array(CLIFFORD_ANGLES).tolist()


def test_angle_to_probabilities():
    for sigma in np.linspace(0.1, 2, 20):
        a = _angle_to_probabilities(CLIFFORD_ANGLES, sigma)
        for probs in a:
            assert isinstance(probs, float)


def test_probabilistic_angles_to_clifford():
    for sigma in np.linspace(0.1, 2, 20):
        a = _probabilistic_angle_to_clifford(CLIFFORD_ANGLES, sigma)
        for ang in a:
            for cliff in CLIFFORD_ANGLES:
                if ang == cliff:
                    check = True
            assert check
