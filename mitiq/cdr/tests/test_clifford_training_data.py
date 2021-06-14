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
from mitiq.cdr.clifford_training_data import (
    _is_clifford_angle,
    is_clifford,
    _map_to_near_clifford,
    _select,
    _replace,
    _closest_clifford,
    _random_clifford,
    _angle_to_proximity,
    _angle_to_proximities,
    _probabilistic_angle_to_clifford,
    count_non_cliffords,
    generate_training_circuits,
    _CLIFFORD_ANGLES,
)
from mitiq.cdr._testing import random_x_z_cnot_circuit


@pytest.mark.parametrize("circuit_type", SUPPORTED_PROGRAM_TYPES.keys())
def test_is_clifford_with_clifford(circuit_type):
    circuit = convert_from_mitiq(
        cirq.Circuit(cirq.Z.on(cirq.LineQubit(0))), circuit_type
    )
    assert is_clifford(circuit)


@pytest.mark.parametrize("circuit_type", SUPPORTED_PROGRAM_TYPES.keys())
def test_is_clifford_with_nonclifford(circuit_type):
    circuit = convert_from_mitiq(
        cirq.Circuit(cirq.T.on(cirq.LineQubit(0))), circuit_type
    )
    assert not is_clifford(circuit)


def test_generate_training_circuits():
    circuit = random_x_z_cnot_circuit(
        cirq.LineQubit.range(3), n_moments=5, random_state=1
    )
    assert not is_clifford(circuit)

    (clifford_circuit,) = generate_training_circuits(
        circuit, num_training_circuits=1, fraction_non_clifford=0.0
    )
    assert is_clifford(clifford_circuit)


@pytest.mark.parametrize("circuit_type", SUPPORTED_PROGRAM_TYPES.keys())
def test_generate_training_circuits_any_qprogram(circuit_type):
    circuit = random_x_z_cnot_circuit(
        cirq.LineQubit.range(3), n_moments=5, random_state=1
    )
    circuit = convert_from_mitiq(circuit, circuit_type)

    (clifford_circuit,) = generate_training_circuits(
        circuit, num_training_circuits=1, fraction_non_clifford=0.0
    )
    assert is_clifford(clifford_circuit)


@pytest.mark.parametrize("method", ("uniform", "gaussian"))
def test_select_all(method):
    q = cirq.LineQubit(0)
    ops = [cirq.ops.rz(0.01).on(q), cirq.ops.rz(-0.77).on(q)]
    indices = _select(
        ops, 0.0, method=method, random_state=np.random.RandomState(1)
    )
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

    new_ops = _replace(
        non_clifford_ops=ops,
        method=method,
        random_state=np.random.RandomState(1),
    )

    assert len(new_ops) == len(ops)
    assert all(cirq.has_stabilizer_effect(op) for op in new_ops)


def test_map_to_near_clifford():
    q = cirq.LineQubit(0)
    ops = [cirq.ops.rz(np.pi / 2.0 + 0.01).on(q), cirq.ops.rz(-0.22).on(q)]

    new_ops = _map_to_near_clifford(
        ops,
        fraction_non_clifford=0.0,
        method_select="uniform",
        method_replace="uniform",
        random_state=np.random.RandomState(2),
    )
    expected_ops = [cirq.rz(np.pi * 1).on(q), cirq.rz(np.pi * 1.5).on(q)]
    assert new_ops == expected_ops


def test_generate_training_circuits_bad_methods():
    with pytest.raises(ValueError):
        generate_training_circuits(
            Circuit(cirq.ops.rx(0.5).on(cirq.LineQubit(0))),
            num_training_circuits=1,
            fraction_non_clifford=0.0,
            method_select="unknown select method",
        )

    with pytest.raises(ValueError):
        generate_training_circuits(
            Circuit(cirq.ops.rx(0.5).on(cirq.LineQubit(0))),
            num_training_circuits=1,
            fraction_non_clifford=0.0,
            method_replace="unknown replace method",
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
@pytest.mark.parametrize(
    "kwargs", [{}, {"sigma_select": 0.5, "sigma_replace": 0.5}]
)
def test_generate_training_circuits_mega(
    method_select, method_replace, kwargs
):
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
        **kwargs,
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

    indices = _select(
        non_clifford_ops,
        fraction_non_clifford,
        method=method,
        sigma=0.5,
        random_state=np.random.RandomState(1),
    )
    assert all(isinstance(index, int) for index in indices)
    assert len(indices) == int(
        round((1.0 - fraction_non_clifford) * len(non_clifford_ops))
    )


@pytest.mark.parametrize("circuit_type", SUPPORTED_PROGRAM_TYPES.keys())
def test_count_non_cliffords(circuit_type):
    a, b = cirq.LineQubit.range(2)
    circuit = Circuit(
        cirq.rz(0.0).on(a),  # Clifford.
        cirq.rx(0.1 * np.pi).on(b),  # Non-Clifford.
        cirq.rx(0.5 * np.pi).on(b),  # Clifford
        cirq.rz(0.4 * np.pi).on(b),  # Non-Clifford.
        cirq.rz(0.5 * np.pi).on(b),  # Clifford.
        cirq.CNOT.on(a, b),  # Clifford.
    )
    circuit = convert_from_mitiq(circuit, circuit_type)

    assert count_non_cliffords(circuit) == 2


def test_count_non_cliffords_empty_circuit():
    assert count_non_cliffords(Circuit()) == 0


def test_is_clifford_angle():
    for p in range(4):
        assert _is_clifford_angle(p * np.array(_CLIFFORD_ANGLES)).all()

    assert not _is_clifford_angle(-0.17)


def test_closest_clifford():
    for ang in _CLIFFORD_ANGLES:
        angs = np.linspace(ang - np.pi / 4 + 0.01, ang + np.pi / 4 - 0.01)
        for a in angs:
            assert _closest_clifford(a) == ang


def test_random_clifford():
    assert set(_random_clifford(20, np.random.RandomState(1))).issubset(
        _CLIFFORD_ANGLES
    )


def test_angle_to_proximities():
    for sigma in np.linspace(0.1, 2, 10):
        for ang in _CLIFFORD_ANGLES:
            probabilities = _angle_to_proximities(ang, sigma)
            assert (isinstance(p, float) for p in probabilities)


def test_angle_to_proximity():
    for sigma in np.linspace(0.1, 2, 10):
        probabilities = _angle_to_proximity(_CLIFFORD_ANGLES, sigma)
        assert all(isinstance(p, float) for p in probabilities)


def test_probabilistic_angles_to_clifford():
    for sigma in np.linspace(0.1, 2, 10):
        angles = _probabilistic_angle_to_clifford(
            _CLIFFORD_ANGLES, sigma, np.random.RandomState(1)
        )
        assert all(a in _CLIFFORD_ANGLES for a in angles)
