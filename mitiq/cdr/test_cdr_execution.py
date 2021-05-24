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
import pytest

import numpy as np

import cirq
from cirq.circuits import Circuit
from cirq import Simulator
from cirq import depolarize
from cirq import DensityMatrixSimulator

from cdr_execution import execute_with_CDR

from mitiq.zne.scaling import (
    fold_gates_from_left,
    fold_gates_from_right,
    fold_gates_at_random,
)

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
    cirq.LineQubit.range(4), n_moments=8, random_state=1
)

sigma_z = np.diag([1, -1])
obs = np.kron(np.identity((8)), sigma_z)
obs = np.diag(obs)

results = execute_with_CDR(
    circuit,
    executor,
    simulator_statevector,
    obs,
    4,
    0.5,
    scale_noise=fold_gates_from_left,
    noise_scaling_factors=3,
)

exact_soln = calculate_observable(
    simulator_statevector(circuit), observable=obs
)


print("Noisy", results[0][0])
print("Mitigated", results[1])
print("exact", exact_soln)
