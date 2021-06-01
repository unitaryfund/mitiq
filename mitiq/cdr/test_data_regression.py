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

"""Tests for data regression code in Clifford data regression."""
from typing import List

import pytest
import numpy as np

import cirq
from cirq.circuits import Circuit
from cirq import Simulator
from cirq import depolarize
from cirq import DensityMatrixSimulator

from mitiq.cdr.data_regression import (
    scale_noise_in_circuits,
    construct_training_data_floats,
    construct_circuit_data_floats,
    linear_fit_function,
    calculate_observable,
    dictionary_to_probabilities,
)

from mitiq.zne.scaling import (
    fold_gates_from_left,
    fold_gates_from_right,
    fold_gates_at_random,
)

from mitiq.cdr.clifford_training_data import generate_training_circuits

from collections import Counter

# Defines circuit used in unit tests:


def random_x_z_circuit(qubits, n_moments, random_state) -> Circuit:
    angles = np.linspace(0.0, 2 * np.pi, 6)
    oneq_gates = [cirq.ops.rz(a) for a in angles]
    oneq_gates.append(cirq.ops.rx(np.pi / 2))
    gate_domain = {oneq_gate: 1 for oneq_gate in oneq_gates}

    return cirq.testing.random_circuit(
        qubits=qubits,
        n_moments=n_moments,
        op_density=1.0,
        gate_domain=gate_domain,
        random_state=random_state,
    )


# Defines executor used in circuit tests:


def executor(circuit: Circuit) -> dict:
    """ executor for unit tests. """
    circuit_copy = circuit.copy()
    for qid in list(Circuit.all_qubits(circuit_copy)):
        circuit_copy.append(cirq.measure(qid))
    simulator = DensityMatrixSimulator()
    shots = 8192
    noise = 0.1
    circuit_with_noise = circuit_copy.with_noise(depolarize(p=noise))
    result = simulator.run(circuit_with_noise, repetitions=shots)
    counts = result.multi_measurement_histogram(
        keys=Circuit.all_qubits(circuit_with_noise)
    )
    dict_counts = counter_to_dict(counts)
    return dict_counts


# Defines a function (which could be user defined) that converts a python
# Counter object which is returned by cirq into a dictionary of counts.
def counter_to_dict(counts: Counter) -> dict:
    """ Returns a dictionary of counts. Takes cirq output 'Counter' object to
    binary counts. I assume this is the format which we will be working with
    from now on.
    Args:
        counts: Counter object returned by cirq with the results of a circuit.
    """
    counts_dict = {}
    for key, value in counts.items():
        key2 = bin(int("".join(str(ele) for ele in key), 2))
        counts_dict[key2] = value
    return counts_dict


def simulator_statevector(circuit: Circuit) -> np.ndarray:
    circuit_copy = circuit.copy()
    simulator = Simulator()
    result = simulator.simulate(circuit_copy)
    statevector = result.final_state_vector
    return statevector


def simulator_counts(circuit: Circuit) -> dict:
    circuit_copy = circuit.copy()
    for qid in list(Circuit.all_qubits(circuit_copy)):
        circuit_copy.append(cirq.measure(qid))
    simulator = DensityMatrixSimulator()
    shots = 8192
    result = simulator.run(circuit_copy, repetitions=shots)
    counts = result.multi_measurement_histogram(
        keys=Circuit.all_qubits(circuit_copy)
    )
    dict_counts = counter_to_dict(counts)
    return dict_counts


# circuit used for unit tests:
circuit = random_x_z_circuit(
    cirq.LineQubit.range(1), n_moments=6, random_state=1
)


# training set used for unit tests:
training_circuits_list = generate_training_circuits(circuit, 3, 0.3)

# some example data used in following tests:
all_training_circuits_list = scale_noise_in_circuits(
    training_circuits_list, fold_gates_from_left, 3
)
training_circuits_raw_data = [
    [] for i in range(len(all_training_circuits_list))
]
# list to store simulated training circuits:
training_circuits_simulated_data = []
for i, training_circuits in enumerate(all_training_circuits_list):
    for j, circuit in enumerate(training_circuits):
        training_circuits_raw_data[i].append(executor(circuit))
        # runs the circuits with no increased noise in the simulator:
        if i == 0:
            training_circuits_simulated_data.append(
                simulator_statevector(circuit)
            )

results_training_circuits = (
    training_circuits_raw_data,
    training_circuits_simulated_data,
)

results_training_circuits_one_noise_level = (
    [training_circuits_raw_data[0]],
    training_circuits_simulated_data,
)

all_circuits_of_interest = scale_noise_in_circuits(
    [circuit], fold_gates_from_left, 3
)

results_circuit_of_interest = []
for circuit_ in all_circuits_of_interest:
    circuit_raw_result = executor(circuit_[0])
    results_circuit_of_interest.append(circuit_raw_result)

results_circuit_of_interest_one_noise_level = [results_circuit_of_interest[0]]

sigma_z = np.diag([1, -1])
sigma_z = np.diag(sigma_z)


@pytest.mark.parametrize(
    "fold_method",
    [fold_gates_from_left, fold_gates_from_right, fold_gates_at_random],
)
@pytest.mark.parametrize("scale_factors", [[3, 5], 3])
def test_scale_noise_in_circuits(fold_method, scale_factors):
    circuits = training_circuits_list
    folded_circuit = scale_noise_in_circuits(
        circuits, fold_method, scale_factors
    )
    if isinstance(circuits, List):
        assert len(folded_circuit[0]) == len(circuits)
        assert len(folded_circuit[1][0]) > len(folded_circuit[0][0])


def test_calculate_observable():
    sim_state = simulator_statevector(circuit)
    sim_counts = simulator_counts(circuit)
    obs_state = calculate_observable(sim_state, sigma_z)
    obs_counts = calculate_observable(sim_counts, sigma_z)
    assert abs(obs_state - obs_counts) <= 0.015


@pytest.mark.parametrize("noise_levels", [1, 2])
def test_construct_training_data_floats(noise_levels):
    if noise_levels == 1:
        train_data = construct_training_data_floats(
            results_training_circuits_one_noise_level, sigma_z
        )
    elif noise_levels == 2:
        train_data = construct_training_data_floats(
            results_training_circuits, sigma_z
        )
    assert len(train_data[0][0]) == noise_levels


@pytest.mark.parametrize("noise_levels", [1, 2])
def test_construct_circuit_data_floats(noise_levels):
    if noise_levels == 1:
        data = construct_circuit_data_floats(
            results_circuit_of_interest_one_noise_level, sigma_z
        )
    elif noise_levels == 2:
        data = construct_circuit_data_floats(
            results_circuit_of_interest, sigma_z
        )
    assert len(data) == noise_levels


def test_linear_fit_function():
    a = np.array([1, 2, 3])
    b = np.array([1, 1, 1, 1])
    point = linear_fit_function(a, b)
    assert point == 7


def test_dictionary_to_probabilities():
    Q = 1
    sim_counts = simulator_counts(circuit)
    state = dictionary_to_probabilities(sim_counts, Q)
    assert bin(0) in state.keys()
    assert bin(1) in state.keys()
    assert (isinstance(i, float) for i in list(state.values()))
