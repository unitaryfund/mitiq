# Copyright (C) 2021 Unitary Fund
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import pytest

import cirq
from cirq.circuits import Circuit
from random import randint, uniform
import numpy as np
from pytest import raises
from clifford_training_data import (
    _is_clifford_angle,
    _is_clifford,
    is_clifford,
    _map_to_near_clifford,
    _select,
    _replace,
    _project_to_closest_clifford,
    _closest_clifford,
    _random_clifford,
    _angle_to_probabilities,
    _probabilistic_angle_to_clifford,
    count_non_cliffords,
    generate_training_circuits,
    _CLIFFORD_ANGLES,
)
from cirq.experiments import (
    random_rotations_between_grid_interaction_layers_circuit,
)
from qiskit import compiler, QuantumCircuit
from mitiq.mitiq_qiskit.conversions import to_qiskit, from_qiskit


"""Tests for training circuits generation for Clifford data regression.
"""


def random_circuit(qubits: int, depth: int,) -> Circuit:
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
    circuit = random_rotations_between_grid_interaction_layers_circuit(
        qubits=qubits, depth=depth, seed=0
    )
    circuit.append(cirq.measure(*qubits, key="z"))
    return circuit


def qiskit_circuit_transpilation(circ: QuantumCircuit,) -> QuantumCircuit:
    """Decomposes qiskit circuit object into Rz, Rx(pi/2) (sx), X and CNOT \
       gates.
    Args:
        circ: original circuit of interest assumed to be qiskit circuit object.
    Returns:
        circ_new: new circuite compiled and decomposed into the above gate set.
    """
    # this decomposes the circuit into u3 and cnot gates:
    circ = compiler.transpile(
        circ, basis_gates=["sx", "rz", "cx", "x"], optimization_level=3
    )
    # print(circ.draw())
    # now for each U3(theta, phi, lambda), this can be converted into
    # Rz(phi+pi)Rx(pi/2)Rz(theta+pi)Rx(pi/2)Rz(lambda)
    circ_new = QuantumCircuit(len(circ.qubits), len(circ.clbits))
    for i in range(len(circ.data)):
        # get information for the gate
        gate = circ.data[i][0]
        name = gate.name
        if name == "cx":
            qubit = [circ.data[i][1][0].index, circ.data[i][1][1].index]
            parameters = []
            circ_new.cx(qubit[0], qubit[1])
        if name == "rz":
            parameters = (float(gate.params[0])) % (2 * np.pi)
            # leave out empty Rz gates:
            if parameters != 0:
                qubit = circ.data[i][1][0].index
                circ_new.rz(parameters, qubit)
        if name == "sx":
            parameters = np.pi / 2
            qubit = circ.data[i][1][0].index
            circ_new.rx(parameters, qubit)
        if name == "x":
            qubit = circ.data[i][1][0].index
            circ_new.x(qubit)
        elif name == "measure":
            qubit = circ.data[i][1][0].index
            cbit = circ.data[i][2][0].index
            circ_new.measure(qubit, cbit)
    return circ_new


num_qubits = 4
layers = 10
num_training_circuits = 40
fraction_non_clifford = 0.5
circuit = cirq.circuits.Circuit(random_circuit(num_qubits, layers))
circuit = from_qiskit(qiskit_circuit_transpilation(to_qiskit(circuit)))
non_cliffords = count_non_cliffords(circuit)


def random_x_z_cnot_circuit(qubits, n_moments, random_state) -> Circuit:
    angles = np.linspace(0.0, 2 * np.pi, 8)
    oneq_gates = [gate(a) for a in angles for gate in
                  (cirq.ops.rx, cirq.ops.rz)]
    gate_domain = {oneq_gate: 1 for oneq_gate in oneq_gates}
    gate_domain.update({cirq.ops.CNOT: 2})

    return cirq.testing.random_circuit(
        qubits=qubits,
        n_moments=n_moments,
        op_density=1.0,
        gate_domain=gate_domain,
        random_state=random_state,
    )


def test_generate_training_circuits():
    circuit = random_x_z_cnot_circuit(
        cirq.LineQubit.range(3), n_moments=5, random_state=1
    )
    assert not is_clifford(circuit)

    clifford_circuit, = generate_training_circuits(
        circuit, num_training_circuits=1, fraction_non_clifford=0.0
    )
    assert is_clifford(clifford_circuit)



# def test_project_to_closest_clifford_with_clifford_ops():
#     ops = [cirq.ops.rz(a).on(cirq.LineQubit(0)) for a in (0, 0.5, 1.0, 1.5)]
#     clifford_ops = _project_to_closest_clifford(ops)
#     print(clifford_ops)
#     assert False


@pytest.mark.parametrize("method", ("uniform", "gaussian"))
def test_select_all(method):
    q = cirq.LineQubit(0)
    ops = [cirq.ops.rz(0.01).on(q), cirq.ops.rz(-0.77).on(q)]
    indices = _select(ops, 0.0, method=method, random_state=np.random.RandomState(1))
    assert np.allclose(indices, np.array(list(range(len(ops)))))


@pytest.mark.parametrize("method", ("uniform", "gaussian"))
def test_select_some(method):
    n = 10  # Number to select.
    q = cirq.GridQubit(1, 1)
    ops = [cirq.ops.rz(a).on(q) for a in np.random.randn(n)]
    indices = _select(ops, fraction_non_clifford=0.5, method=method)
    assert len(indices) == n // 2


def test_select_bad_method():
    with pytest.raises(ValueError, match="Arg `method_select` must be"):
        _select([], fraction_non_clifford=0.0, method="unknown method")


@pytest.mark.parametrize("method", ("closest", "uniform", "gaussian"))
def test_replace(method):
    q = cirq.LineQubit(0)
    ops = [cirq.ops.rz(0.01).on(q), cirq.ops.rz(-0.77).on(q)]

    new_ops = _replace(non_clifford_ops=ops, method=method, random_state=np.random.RandomState(1))

    assert len(new_ops) == len(ops)
    assert all(_is_clifford(op) for op in new_ops)


def test_map_to_near_clifford():
    q = cirq.LineQubit(0)
    ops = [cirq.ops.rz(0.01).on(q), cirq.ops.rz(-0.77).on(q)]

    new_ops = _map_to_near_clifford(
        ops,
        fraction_non_clifford=0.0,
        method_select="uniform",
        method_replace="uniform",
        seed=1,
    )
    print(new_ops)

    assert False


def test_generate_training_circuits_bad_methods():
    with pytest.raises(ValueError):
        generate_training_circuits(
            Circuit(cirq.ops.rx(0.5).on(cirq.LineQubit(0))),
            num_training_circuits=1,
            fraction_non_clifford=0.0,
            method_select="unknown select method"
        )

    with pytest.raises(ValueError):
        generate_training_circuits(
            Circuit(cirq.ops.rx(0.5).on(cirq.LineQubit(0))),
            num_training_circuits=1,
            fraction_non_clifford=0.0,
            method_replace="unknown replace method"
        )


def test_generate_training_circuits_with_clifford_circuit():
    with pytest.raises(ValueError, match="Circuit is already Clifford."):
        generate_training_circuits(
            Circuit(cirq.ops.rx(0.0).on(cirq.LineQubit(0))),
            num_training_circuits=1,
            fraction_non_clifford=0.0,
        )

# def test_generate_training_circuits_mega():
#     """Test that generate_training_circuits function is working properly with
#     the random projection method.
#     """
#     method_select_options_list = ["uniform", "gaussian"]
#     method_replace_options_list = ["uniform", "gaussian", "closest"]
#     additional_options = {"sigma_select": 0.5, "sigma_replace": 0.5}
#     non_cliffords = count_non_cliffords(circuit)
#     random_state = 13
#
#     for method_select in method_select_options_list:
#         for method_replace in method_replace_options_list:
#             test_training_set_circuits = generate_training_circuits(
#                 circuit,
#                 num_training_circuits,
#                 fraction_non_clifford,
#                 method_select,
#                 method_replace,
#             )
#             test_training_circuits_with_options = generate_training_circuits(
#                 circuit,
#                 num_training_circuits,
#                 fraction_non_clifford,
#                 method_select,
#                 method_replace,
#                 random_state,
#                 kwargs=additional_options,
#             )
#             assert len(test_training_set_circuits) == num_training_circuits
#
#             assert (
#                 len(test_training_circuits_with_options)
#                 == num_training_circuits
#             )
#
#             for i in range(num_training_circuits):
#                 assert count_non_cliffords(
#                     test_training_set_circuits[i]
#                 ) == int(fraction_non_clifford * non_cliffords)
#                 assert len(test_training_set_circuits[i]) == len(circuit)
#                 assert len(test_training_set_circuits[i].all_qubits()) == len(
#                     circuit.all_qubits()
#                 )
#                 assert count_non_cliffords(
#                     test_training_circuits_with_options[i]
#                 ) == int(fraction_non_clifford * non_cliffords)
#                 assert len(test_training_circuits_with_options[i]) == len(
#                     circuit
#                 )
#                 assert len(
#                     test_training_circuits_with_options[i].all_qubits()
#                 ) == len(circuit.all_qubits())

#
# def test_map_to_near_cliffords():
#     method_select_options_list = ["uniform", "gaussian"]
#     method_replace_options_list = ["uniform", "gaussian", "closest"]
#     additional_options = {"sigma_select": 0.5, "sigma_replace": 0.5}
#     non_cliffords = count_non_cliffords(circuit)
#     random_state = np.random.RandomState(1)
#     operations = np.array(list(circuit.all_operations()))
#     gates = np.array([op.gate for op in operations])
#     qubits = np.array([op.qubits[0] for op in operations])
#     positions = np.array(range(0, len(gates)))
#     zgatesmask = np.array(
#         [isinstance(gate, cirq.ops.common_gates.ZPowGate) for gate in gates]
#     )
#     r_z_gates = operations[zgatesmask]
#     r_z_positions = positions[zgatesmask]
#     r_z_qubits = qubits[zgatesmask]
#     angles = angles = np.array([op.gate.exponent * np.pi for op in r_z_gates])
#     mask_non_clifford = ~_is_clifford_angle(angles)
#     rz_non_clifford = angles[mask_non_clifford]
#     position_non_clifford = r_z_positions[mask_non_clifford]
#     qubits_non_cliff = r_z_qubits[mask_non_clifford]
#     for method_select in method_select_options_list:
#         for method_replace in method_replace_options_list:
#             projected_circuit = _map_to_near_clifford(
#                 operations.copy(),
#                 rz_non_clifford.copy(),
#                 position_non_clifford.copy(),
#                 qubits_non_cliff.copy(),
#                 fraction_non_clifford,
#                 random_state,
#                 method_select,
#                 method_replace,
#             )
#             projected_circuit_with_options = _map_to_near_clifford(
#                 operations.copy(),
#                 rz_non_clifford.copy(),
#                 position_non_clifford.copy(),
#                 qubits_non_cliff.copy(),
#                 fraction_non_clifford,
#                 random_state,
#                 method_select,
#                 method_replace,
#                 kwargs=additional_options,
#             )
#             assert count_non_cliffords(projected_circuit) == int(
#                 fraction_non_clifford * non_cliffords
#             )
#             assert len(projected_circuit) == len(circuit)
#             assert len(projected_circuit.all_qubits()) == len(
#                 circuit.all_qubits()
#             )
#             assert count_non_cliffords(projected_circuit_with_options) == int(
#                 fraction_non_clifford * non_cliffords
#             )
#             assert len(projected_circuit_with_options) == len(circuit)
#             assert len(projected_circuit_with_options.all_qubits()) == len(
#                 circuit.all_qubits()
#             )
#
#
# def test_select():
#     method_select_options_list = ["uniform", "gaussian"]
#     non_cliffords = count_non_cliffords(circuit)
#     operations = np.array(list(circuit.all_operations()))
#     gates = np.array([op.gate for op in operations])
#     rzgatemask = np.array(
#         [isinstance(i, cirq.ops.common_gates.ZPowGate) for i in gates]
#     )
#     r_z_gates = operations[rzgatemask]
#     angles = np.array([op.gate.exponent * np.pi for op in r_z_gates])
#     mask_non_clifford = ~_is_clifford_angle(angles)
#     rz_non_clifford = angles[mask_non_clifford]
#     rz_non_cliff_copy = rz_non_clifford.copy()
#     random_state = np.random.RandomState(1)
#     sigma_select = 0.5
#     for method_select in method_select_options_list:
#         columns_to_change = _select(
#             rz_non_cliff_copy,
#             fraction_non_clifford,
#             method_select,
#             sigma_select,
#             random_state,
#         )
#         assert len(columns_to_change) == (
#             non_cliffords - int(non_cliffords * fraction_non_clifford)
#         )
#
#
# def test_replace():
#     method_select_options_list = ["uniform", "gaussian"]
#     method_replace_options_list = ["uniform", "gaussian", "closest"]
#     non_cliffords = count_non_cliffords(circuit)
#     operations = np.array(list(circuit.all_operations()))
#     gates = np.array([op.gate for op in operations])
#     mask = np.array(
#         [isinstance(i, cirq.ops.common_gates.ZPowGate) for i in gates]
#     )
#     r_z_gates = operations[mask]
#     angles = np.array([op.gate.exponent * np.pi for op in r_z_gates])
#     mask_non_clifford = ~_is_clifford_angle(angles)
#     rz_non_clifford = angles[mask_non_clifford]
#     rz_non_cliff_copy = rz_non_clifford.copy()
#     sigma_select = 0.5
#     sigma_replace = 0.5
#     random_state = np.random.RandomState(1)
#     for method_select in method_select_options_list:
#         for method_replace in method_replace_options_list:
#             columns_to_change = _select(
#                 rz_non_cliff_copy,
#                 fraction_non_clifford,
#                 method_select,
#                 sigma_select,
#                 random_state,
#             )
#             rz_non_cliff_selected = rz_non_cliff_copy[columns_to_change]
#             rz_non_cliff_replaced = _replace(
#                 rz_non_cliff_selected,
#                 method_replace,
#                 sigma_replace,
#                 random_state,
#             )
#             assert _is_clifford_angle(rz_non_cliff_replaced.all())
#             assert len(rz_non_cliff_replaced) == (
#                 non_cliffords - int(non_cliffords * fraction_non_clifford)
#             )
#
#
# def test_get_argument():
#     operations = np.array(list(circuit.all_operations()))
#     gates = np.array([op.gate for op in operations])
#     mask = np.array(
#         [isinstance(i, cirq.ops.common_gates.ZPowGate) for i in gates]
#     )
#     r_z_gates = operations[mask]
#     args = np.array([op.gate.exponent * np.pi for op in r_z_gates])
#     for arg in args:
#         assert type(arg) == np.float64
#
#
# def test_count_non_cliffords():
#     number_non_cliffords = 0
#     example_circuit = QuantumCircuit(1)
#     for i in range(100):
#         rand = randint(1, 2)
#         rand2 = randint(1, 4) - 1
#         if rand % 2 == 0:
#             example_circuit.rz(_CLIFFORD_ANGLES[rand2], 0)
#         else:
#             example_circuit.rz(uniform(0, 2 * np.pi), 0)
#             number_non_cliffords += 1
#         example_circuit = from_qiskit(example_circuit)
#         assert count_non_cliffords(example_circuit) == number_non_cliffords
#         example_circuit = to_qiskit(example_circuit)
#
#
# def test_is_clifford_angle():
#     cliff_angs = np.array(_CLIFFORD_ANGLES)
#
#     for i in range(15):
#         assert _is_clifford_angle(int(i) * cliff_angs).all()
#         ang = uniform(0, 2 * np.pi)
#         assert not _is_clifford_angle(ang)
#
#
# def test_closest_clifford():
#     for ang in _CLIFFORD_ANGLES:
#         angs = np.linspace(ang - np.pi / 4 + 0.01, ang + np.pi / 4 - 0.01)
#         for a in angs:
#             assert _closest_clifford(a) == ang
#
#
# def test_random_clifford():
#     assert set(
#         _random_clifford(20, np.random.RandomState(1))
#     ).issubset(_CLIFFORD_ANGLES)
#
#
# def test_angle_to_probabilities():
#     for sigma in np.linspace(0.1, 2, 20):
#         a = _angle_to_probabilities(_CLIFFORD_ANGLES, sigma)
#         for probs in a:
#             assert isinstance(probs, float)
#
#
# def test_probabilistic_angles_to_clifford():
#     random_state = np.random.RandomState(1)
#     for sigma in np.linspace(0.1, 2, 20):
#         a = _probabilistic_angle_to_clifford(
#             _CLIFFORD_ANGLES, sigma, random_state
#         )
#         for ang in a:
#             for cliff in _CLIFFORD_ANGLES:
#                 if ang == cliff:
#                     check = True
#             assert check
