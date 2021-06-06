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
import pytest
import numpy as np

import cirq

from mitiq.cdr.execute import (
    construct_training_data_floats,
    calculate_observable,
    measurements_to_probabilities,
)
from mitiq.cdr._testing import random_x_z_circuit, executor, simulator_statevector, simulator
from mitiq.cdr.clifford_training_data import generate_training_circuits
from mitiq.zne.scaling import fold_gates_from_left

# Observables.
sigma_z = np.diag(np.diag([1, -1]))


# Test circuit.
test_circuit = random_x_z_circuit(
    cirq.LineQubit.range(1), n_moments=6, random_state=1
)
training_circuits = generate_training_circuits(test_circuit, 3, 0.3)

# some example data used in following tests:
all_training_circuits_list = [
    [fold_gates_from_left(c, s) for c in training_circuits]
    for s in (1, 3)
]

training_circuits_raw_data = [
    [] for i in range(len(all_training_circuits_list))
]
# list to store simulated training circuits:
training_circuits_simulated_data = []
for i, tc in enumerate(all_training_circuits_list):
    for j, test_circuit in enumerate(tc):
        training_circuits_raw_data[i].append(executor(test_circuit))
        # runs the circuits with no increased noise in the simulator:
        if i == 0:
            training_circuits_simulated_data.append(
                simulator_statevector(test_circuit)
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
    [fold_gates_from_left(c, s) for c in [test_circuit]] for s in (1, 3)
]

results_circuit_of_interest = []
for circuit_ in all_circuits_of_interest:
    circuit_raw_result = executor(circuit_[0])
    results_circuit_of_interest.append(circuit_raw_result)

results_circuit_of_interest_one_noise_level = [results_circuit_of_interest[0]]


@pytest.mark.parametrize("op_and_expectation_value", ((cirq.I, 1.0), (cirq.H, 0.0), (cirq.X, -1.0)))
def test_calculate_observable_sigmaz(op_and_expectation_value):
    """Tests <psi|Z|psi> is correct for |psi> \in {|0>, |+>, |1>}."""
    op, expected = op_and_expectation_value
    circuit = cirq.Circuit(op.on(cirq.LineQubit(0)))
    assert np.isclose(
        calculate_observable(simulator_statevector(circuit), sigma_z), expected,
        atol=1e-7
    )

    assert np.isclose(
        calculate_observable(simulator(circuit, shots=10_000), sigma_z), expected,
        atol=1e-2
    )


@pytest.mark.parametrize("noise_levels", [1, 2])
def test_construct_training_data_floats(noise_levels):
    results = (
        results_training_circuits_one_noise_level
        if noise_levels == 1
        else results_training_circuits
    )
    train_data = construct_training_data_floats(results, sigma_z)
    print(train_data)
    # assert False
    assert len(train_data[0][0]) == noise_levels


def test_dictionary_to_probabilities():
    counts = {bin(0): 2, bin(1): 3}
    normalized_counts = measurements_to_probabilities(counts, nqubits=1)
    assert normalized_counts == {bin(0): 2 / 5, bin(1): 3 / 5}


def test_dictionary_to_probabilities_1():
    sim_counts = simulator(test_circuit)
    state = measurements_to_probabilities(sim_counts, nqubits=1)
    assert bin(0) in state.keys()
    assert bin(1) in state.keys()
    assert (isinstance(i, float) for i in list(state.values()))
