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

"""Qiskit utility functions."""
import numpy as np
import copy
import qiskit
from qiskit import QuantumCircuit
from typing import Optional

# Noise simulation packages
from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer.noise.errors.standard_errors import (
    depolarizing_error,
)

QASM_SIMULATOR = qiskit.Aer.get_backend("qasm_simulator")
WVF_SIMULATOR = qiskit.Aer.get_backend("statevector_simulator")


def initialized_depolarizing_noise(noise: float) -> NoiseModel:
    """Initializes a depolarizing noise Qiskit NoiseModel.

    Args:
        noise: The depolarizing noise strength as a float, i.e. 0.001 is 0.1%.

    Returns:
        A Qiskit depolarizing NoiseModel.
    """
    # initialize a qiskit noise model
    noise_model = NoiseModel()

    # we assume the same depolarizing error for each
    # gate of the standard IBM basis
    noise_model.add_all_qubit_quantum_error(
        depolarizing_error(noise, 1), ["u1", "u2", "u3"]
    )
    noise_model.add_all_qubit_quantum_error(
        depolarizing_error(noise, 2), ["cx"]
    )
    return noise_model


def execute(circ: QuantumCircuit, obs: np.ndarray) -> float:
    """Simulates noiseless wavefunction evolution and returns the
    expectation value of some observable.

    Args:
        circ: The input Qiskit circuit.
        obs: The observable to measure as a NumPy array.

    Returns:
        The expectation value of obs as a float.
    """
    result = qiskit.execute(circ, WVF_SIMULATOR).result()
    final_wvf = result.get_statevector()
    return np.real(final_wvf.conj().T @ obs @ final_wvf)


def execute_with_shots(
    circ: QuantumCircuit, obs: np.ndarray, shots: int
) -> float:
    """Simulates the evolution of the circuit and returns
    the expectation value of the observable.

    Args:
        circ: The input Qiskit circuit.
        obs: The observable to measure as a NumPy array.
        shots: The number of measurements.

    Returns:
        The expectation value of obs as a float.

    """
    circ = copy.deepcopy(circ)
    # we need to modify the circuit to measure obs in its eigenbasis
    # we do this by appending a unitary operation
    # obtains a U s.t. obs = U diag(eigvals) U^dag
    eigvals, U = np.linalg.eigh(obs)
    circ.unitary(np.linalg.inv(U), qubits=range(circ.num_qubits))

    circ.measure_all()

    # execution of the experiment
    job = qiskit.execute(
        circ,
        backend=QASM_SIMULATOR,
        # we want all gates to be actually applied,
        # so we skip any circuit optimization
        optimization_level=0,
        shots=shots,
    )
    results = job.result()
    counts = results.get_counts()
    expectation = 0

    for bitstring, count in counts.items():
        expectation += (
            eigvals[int(bitstring[0 : circ.num_qubits], 2)] * count / shots
        )
    return expectation


def execute_with_noise(
    circ: QuantumCircuit, obs: np.ndarray, noise_model: NoiseModel
) -> float:
    """Simulates the evolution of the noisy circuit and returns
    the expectation value of the observable.

    Args:
        circ: The input Qiskit circuit.
        obs: The observable to measure as a NumPy array.
        noise_model: The input Qiskit noise model.

    Returns:
        The expectation value of obs as a float.
    """
    circ.snapshot("final", snapshot_type="density_matrix")

    # execution of the experiment
    job = qiskit.execute(
        circ,
        backend=QASM_SIMULATOR,
        backend_options={"method": "density_matrix"},
        noise_model=noise_model,
        # we want all gates to be actually applied,
        # so we skip any circuit optimization
        basis_gates=noise_model.basis_gates,
        optimization_level=0,
        shots=1,
    )
    result = job.result()
    rho = result.data()["snapshots"]["density_matrix"]["final"][0]["value"]

    expectation = np.real(np.trace(rho @ obs))
    return expectation


def execute_with_shots_and_noise(
    circ: QuantumCircuit,
    obs: np.ndarray,
    noise_model: NoiseModel,
    shots: int,
    seed: Optional[int] = None,
) -> float:
    """Simulates the evolution of the noisy circuit and returns
    the expectation value of the observable.

    Args:
        circ: The input Qiskit circuit.
        obs: The observable to measure as a NumPy array.
        noise: The input Qiskit noise model.
        shots: The number of measurements.
        seed: Optional seed for qiskit simulator.

    Returns:
        The expectation value of obs as a float.
    """
    circ = copy.deepcopy(circ)
    # we need to modify the circuit to measure obs in its eigenbasis
    # we do this by appending a unitary operation
    # obtains a U s.t. obs = U diag(eigvals) U^dag
    eigvals, U = np.linalg.eigh(obs)
    circ.unitary(np.linalg.inv(U), qubits=range(circ.num_qubits))

    circ.measure_all()

    # execution of the experiment
    job = qiskit.execute(
        circ,
        backend=QASM_SIMULATOR,
        backend_options={"method": "density_matrix"},
        noise_model=noise_model,
        # we want all gates to be actually applied,
        # so we skip any circuit optimization
        basis_gates=noise_model.basis_gates,
        optimization_level=0,
        shots=shots,
        seed_simulator=seed,
    )
    counts = job.result().get_counts()
    expectation = 0

    for bitstring, count in counts.items():
        expectation += (
            eigvals[int(bitstring[0 : circ.num_qubits], 2)] * count / shots
        )
    return expectation
