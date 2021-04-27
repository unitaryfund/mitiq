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

"""Tests for generating (near) Clifford circuits."""
import pytest
import numpy as np

import cirq
from cirq.circuits import Circuit
from random import randint, uniform

from mitiq.cdr.clifford_training_data import (
    _is_clifford_angle,
    _is_clifford,
    is_clifford,
    _map_to_near_clifford,
    _select,
    _replace,
    _closest_clifford,
    _random_clifford,
    _angle_to_probabilities,
    _probabilistic_angle_to_clifford,
    count_non_cliffords,
    generate_training_circuits,
    _CLIFFORD_ANGLES,
)


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
    ops = [cirq.ops.rz(np.pi / 2.0 + 0.01).on(q), cirq.ops.rz(-0.22).on(q)]

    new_ops = _map_to_near_clifford(
        ops,
        fraction_non_clifford=0.0,
        method_select="uniform",
        method_replace="uniform",
        seed=1,
    )
    expected_ops = [cirq.rz(np.pi * 0.5).on(q), cirq.rz(0.0).on(q)]
    assert new_ops == expected_ops


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


@pytest.mark.parametrize("method_select", ["uniform", "gaussian"])
@pytest.mark.parametrize("method_replace", ["uniform", "gaussian", "closest"])
@pytest.mark.parametrize("kwargs", [{}, {"sigma_select": 0.5, "sigma_replace": 0.5}])
def test_generate_training_circuits_mega(method_select, method_replace, kwargs):
    circuit = random_x_z_cnot_circuit(qubits=4, n_moments=10, random_state=1)
    num_train = 10
    fraction_non_clifford = 0.1

    train_circuits = generate_training_circuits(
        circuit,
        num_training_circuits=num_train,
        fraction_non_clifford=0.1,
        random_state=np.random.RandomState(13),
        method_select=method_select,
        method_replace=method_replace,
        **kwargs
    )
    assert len(train_circuits) == num_train

    for train_circuit in train_circuits:
        assert set(train_circuit.all_qubits()) == set(circuit.all_qubits())
        assert count_non_cliffords(
            train_circuit
        ) == int(fraction_non_clifford * count_non_cliffords(circuit))


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

def test_probabilistic_angles_to_clifford():
    for sigma in np.linspace(0.1, 2, 10):
        angles = _probabilistic_angle_to_clifford(
            _CLIFFORD_ANGLES, sigma, np.random.RandomState(1)
        )
        assert all(a in _CLIFFORD_ANGLES for a in angles)

