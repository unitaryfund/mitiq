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

from mitiq._typing import SUPPORTED_PROGRAM_TYPES
from mitiq.interface import convert_from_mitiq
from mitiq.cdr.clifford_utils import (
    is_clifford_angle,
    is_clifford,
    closest_clifford,
    random_clifford,
    angle_to_proximity,
    angle_to_proximities,
    probabilistic_angle_to_clifford,
    count_non_cliffords,
    _CLIFFORD_ANGLES,
)
from mitiq.cdr.random_circuit_generator import (
    _GAUSSIAN,
    RandomCircuitGenerator,
)
from mitiq.cdr._testing import random_x_z_cnot_circuit


def test_generate_training_circuits():
    circuit = random_x_z_cnot_circuit(
        cirq.LineQubit.range(3), n_moments=5, random_state=1
    )
    assert not is_clifford(circuit)

    basic_generator = RandomCircuitGenerator(
        fraction_non_clifford=0.0
    )
    (clifford_circuit,) = basic_generator.generate_circuits(
        circuit, num_circuits_to_generate=1,
    )
    assert is_clifford(clifford_circuit)


@pytest.mark.parametrize("circuit_type", SUPPORTED_PROGRAM_TYPES.keys())
def test_generate_training_circuits_any_qprogram(circuit_type):
    circuit = random_x_z_cnot_circuit(
        cirq.LineQubit.range(3), n_moments=5, random_state=1
    )
    circuit = convert_from_mitiq(circuit, circuit_type)

    basic_generator = RandomCircuitGenerator(
        fraction_non_clifford=0.0
    )
    (clifford_circuit,) = basic_generator.generate_circuits(
        circuit, num_circuits_to_generate=1,
    )
    assert is_clifford(clifford_circuit)


@pytest.mark.parametrize("method", ("uniform", "gaussian"))
def test_select_all(method):
    q = cirq.LineQubit(0)
    ops = [cirq.ops.rz(0.01).on(q), cirq.ops.rz(-0.77).on(q)]
    basic_generator = RandomCircuitGenerator(
        fraction_non_clifford=0.0,
        method_select=method,
        random_state=np.random.RandomState(1),
    )
    indices = basic_generator._select(ops)
    assert np.allclose(indices, np.array(list(range(len(ops)))))


@pytest.mark.parametrize("method", ("uniform", "gaussian"))
def test_select_some(method):
    n = 10  # Number to select.
    q = cirq.GridQubit(1, 1)
    basic_generator = RandomCircuitGenerator(
        fraction_non_clifford=0.5,
        method_select=method,
    )
    ops = [cirq.ops.rz(a).on(q) for a in np.random.randn(n)]
    indices = basic_generator._select(ops)
    assert len(indices) == n // 2


def test_select_bad_method():
    basic_generator = RandomCircuitGenerator(
        fraction_non_clifford=0.0,
        method_select="unknown method",
    )
    with pytest.raises(ValueError, match="Arg `method_select` must be"):
        basic_generator._select([])


@pytest.mark.parametrize("method", ("closest", "uniform", "gaussian"))
def test_swap_operations(method):
    q = cirq.LineQubit(0)
    ops = [cirq.ops.rz(0.01).on(q), cirq.ops.rz(-0.77).on(q)]

    basic_generator = RandomCircuitGenerator(
        fraction_non_clifford=0.0,
        method_replace=method,
        random_state=np.random.RandomState(1),
    )
    new_ops = basic_generator._swap_operations(ops)
    assert len(new_ops) == len(ops)
    assert all(cirq.has_stabilizer_effect(op) for op in new_ops)


def test_map_to_near_clifford():
    q = cirq.LineQubit(0)
    ops = [cirq.ops.rz(np.pi / 2.0 + 0.01).on(q), cirq.ops.rz(-0.22).on(q)]

    random_circuit_generator = RandomCircuitGenerator(
        fraction_non_clifford=0.0, 
        method_select="uniform",
        method_replace="uniform",
        random_state=np.random.RandomState(2),
    )
    new_ops = random_circuit_generator._map_to_near_clifford(ops)
    expected_ops = [cirq.rz(np.pi * 1).on(q), cirq.rz(np.pi * 1.5).on(q)]
    assert new_ops == expected_ops


def test_generate_training_circuits_bad_methods():
    with pytest.raises(ValueError):
        random_circuit_generator = RandomCircuitGenerator(
            fraction_non_clifford=0.0,
            method_select="unknown select method",
        )
        random_circuit_generator.generate_circuits(
            circuit=Circuit(cirq.ops.rx(0.5).on(cirq.LineQubit(0))),
            num_circuits_to_generate=3,
        )

    with pytest.raises(ValueError):
        random_circuit_generator = RandomCircuitGenerator(
            fraction_non_clifford=0.0,
            method_replace="unknown replace method",
        )
        random_circuit_generator.generate_circuits(
            circuit=Circuit(cirq.ops.rx(0.5).on(cirq.LineQubit(0))),
            num_circuits_to_generate=3,
        )


def test_generate_training_circuits_with_clifford_circuit():
    circuit = Circuit(cirq.ops.rx(0.0).on(cirq.LineQubit(0)))
    basic_generator = RandomCircuitGenerator(
        fraction_non_clifford=0.0
    )
    assert basic_generator.generate_circuits(
        circuit, num_circuits_to_generate=2,
    ) == [circuit, circuit]


@pytest.mark.parametrize("method_select", ["uniform", "gaussian"])
@pytest.mark.parametrize("method_replace", ["uniform", "gaussian", "closest"])
@pytest.mark.parametrize(
    "kwargs", [{}, {"sigma_select": 0.5, "sigma_replace": 0.5}]
)
def test_generate_training_circuits_mega(
    method_select, method_replace, kwargs
):
    circuit = random_x_z_cnot_circuit(qubits=4, n_moments=10, random_state=1)
    num_train = 10
    fraction_non_clifford = 0.1

    basic_generator = RandomCircuitGenerator(
        fraction_non_clifford=0.1,
        method_select=method_select,
        method_replace=method_replace,
        random_state=np.random.RandomState(13),
    )
    if _GAUSSIAN == method_select:
        basic_generator.configure_gaussian(
            kwargs.get("sigma_select", 0.5), kwargs.get("sigma_replace", 0.5)
        )
    train_circuits = basic_generator.generate_circuits(
        circuit,
        num_circuits_to_generate=num_train,
    )
    assert len(train_circuits) == num_train

    for train_circuit in train_circuits:
        assert set(train_circuit.all_qubits()) == set(circuit.all_qubits())
        assert count_non_cliffords(train_circuit) == int(
            round(fraction_non_clifford * count_non_cliffords(circuit))
        )


@pytest.mark.parametrize("method", ["uniform", "gaussian"])
def test_select(method):
    q = cirq.NamedQubit("q")
    non_clifford_ops = [
        cirq.ops.rz(a).on(q) for a in np.linspace(1, 2, 10) / np.e
    ]
    fraction_non_clifford = 0.4

    basic_generator = RandomCircuitGenerator(
        fraction_non_clifford=fraction_non_clifford,
        method_select=method,
        random_state=np.random.RandomState(1),
    )
    if method == _GAUSSIAN:
        basic_generator.configure_gaussian(0.5, 0.5)
    indices = basic_generator._select(non_clifford_ops)
    assert all(isinstance(index, int) for index in indices)
    assert len(indices) == int(
        round((1.0 - fraction_non_clifford) * len(non_clifford_ops))
    )
