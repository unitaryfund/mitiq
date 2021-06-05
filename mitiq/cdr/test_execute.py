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

"""Tests for the data regression portion of Clifford data regression."""
from collections import Counter
import pytest
import numpy as np

import cirq
from cirq.circuits import Circuit
from cirq import Simulator
from cirq import depolarize
from cirq import DensityMatrixSimulator

from mitiq.cdr.execute import (
    construct_training_data_floats,
    construct_circuit_data_floats,
    calculate_observable,
    dictionary_to_probabilities,
)
from mitiq.cdr._testing import random_x_z_circuit
from mitiq.cdr.clifford_training_data import generate_training_circuits
from mitiq.zne.scaling import fold_gates_from_left


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


# Executors. TODO: Why aren't these imported from mitiq.mitiq_cirq.cirq_utils?
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
all_training_circuits_list = [
    [fold_gates_from_left(c, s) for c in training_circuits_list]
    for s in (1, 3)
]

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
    training_circuits_simulated_data,
    training_circuits_raw_data,
)

results_training_circuits_one_noise_level = (
    training_circuits_simulated_data,
    [training_circuits_raw_data[0]],
)

all_circuits_of_interest = [
    [fold_gates_from_left(c, s) for c in [circuit]]
    for s in (1, 3)
]

results_circuit_of_interest = []
for circuit_ in all_circuits_of_interest:
    circuit_raw_result = executor(circuit_[0])
    results_circuit_of_interest.append(circuit_raw_result)

results_circuit_of_interest_one_noise_level = [results_circuit_of_interest[0]]

sigma_z = np.diag([1, -1])
sigma_z = np.diag(sigma_z)


def test_calculate_observable():
    sim_state = simulator_statevector(circuit)
    sim_counts = simulator_counts(circuit)
    obs_state = calculate_observable(sim_state, sigma_z)
    obs_counts = calculate_observable(sim_counts, sigma_z)
    assert abs(obs_state - obs_counts) <= 0.015


@pytest.mark.parametrize("noise_levels", [1, 2])
def test_construct_training_data_floats(noise_levels):
    results = results_training_circuits_one_noise_level if noise_levels == 1 else results_training_circuits
    train_data = construct_training_data_floats(results, sigma_z)
    assert len(train_data[0][0]) == noise_levels


@pytest.mark.parametrize("noise_levels", [1, 2])
def test_construct_circuit_data_floats(noise_levels):
    results = results_circuit_of_interest_one_noise_level if noise_levels == 1 else results_circuit_of_interest
    data = construct_circuit_data_floats(results, sigma_z)
    assert len(data) == noise_levels


def test_dictionary_to_probabilities():
    sim_counts = simulator_counts(circuit)
    state = dictionary_to_probabilities(sim_counts, nqubits=1)
    assert bin(0) in state.keys()
    assert bin(1) in state.keys()
    assert (isinstance(i, float) for i in list(state.values()))
