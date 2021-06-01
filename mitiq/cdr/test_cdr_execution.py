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
"""Test code for execution code in Clifford data regression."""
import numpy as np

import cirq
from cirq.circuits import Circuit
from cirq import Simulator
from cirq import depolarize
from cirq import DensityMatrixSimulator

from mitiq.cdr.cdr_execution import execute_with_CDR

from mitiq.zne.scaling import fold_gates_from_left

from mitiq.cdr.data_regression import calculate_observable

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
    # print(circuit)
    circuit_copy = circuit.copy()
    for qid in list(Circuit.all_qubits(circuit_copy)):
        circuit_copy.append(cirq.measure(qid))
    simulator = DensityMatrixSimulator()
    shots = 8192
    noise = 0.5
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


# circuit used for unit tests:
circuit = random_x_z_circuit(
    cirq.LineQubit.range(2), n_moments=2, random_state=1
)

# define observables for testing
sigma_z = np.diag([1, -1])
obs = np.kron(np.identity((2)), sigma_z)
obs2 = np.kron(sigma_z, sigma_z)
obs_list = [np.diag(obs), np.diag(obs2)]

# get exact solution:
exact_solution = []
for obs in obs_list:
    exact_solution.append(
        calculate_observable(simulator_statevector(circuit), observable=obs)
    )


# example fit function used in testing:
def linear_fit_function_no_intercept(X_data, params) -> float:
    return sum(a * x for a, x in zip(params, X_data))


def test_execute_with_cdr():
    kwargs = {
        "method_select": "gaussian",
        "method_replace": "gaussian",
        "sigma_select": 0.5,
        "sigma_replace": 0.5,
        "random_state": 1,
    }
    num_circuits = 4
    frac_non_cliff = 0.5
    results0 = execute_with_CDR(
        circuit,
        executor,
        simulator_statevector,
        obs_list,
        num_circuits,
        frac_non_cliff,
    )
    results1 = execute_with_CDR(
        circuit,
        executor,
        simulator_statevector,
        obs_list,
        num_circuits,
        frac_non_cliff,
        ansatz=linear_fit_function_no_intercept,
        num_parameters=2,
        scale_noise=fold_gates_from_left,
        scale_factors=[3, 5],
        **kwargs,
    )
    results2 = execute_with_CDR(
        circuit,
        executor,
        simulator_statevector,
        obs_list,
        num_circuits,
        frac_non_cliff,
        scale_factors=[3]
    )
    for results in [results0, results1, results2]:
        for i in range(len(results[1])):
            assert abs(results[0][i][0] - exact_solution[i]) > abs(
                results[1][i] - exact_solution[i]
            )
