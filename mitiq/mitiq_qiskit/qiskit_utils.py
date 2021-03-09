# Copyright (C) 2020 Unitary Fund
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
from typing import Optional
from qiskit import QuantumCircuit

# Noise simulation packages
from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer.noise.errors.standard_errors import (
    depolarizing_error,
)

from mitiq.benchmarks.randomized_benchmarking import generate_rb_circuits
from mitiq.mitiq_qiskit.conversions import to_qiskit

QASM_SIMULATOR = qiskit.Aer.get_backend("qasm_simulator")
WVF_SIMULATOR = qiskit.Aer.get_backend("statevector_simulator")


def random_one_qubit_identity_circuit(num_cliffords: int) -> QuantumCircuit:
    """Returns a single-qubit identity circuit.

    Args:
        num_cliffords (int): Number of cliffords used to generate the circuit.

    Returns:
        circuit: Quantum circuit as a :class:`qiskit.QuantumCircuit` object.
    """
    return to_qiskit(
        *generate_rb_circuits(
            n_qubits=1, num_cliffords=num_cliffords, trials=1
        )
    )


def run_with_noise(
    circuit: QuantumCircuit,
    noise: float,
    shots: int,
    seed: Optional[int] = None,
) -> float:
    """Runs the quantum circuit with a depolarizing channel noise model.
    Args:
        circuit: Ideal quantum circuit.
        noise: Noise constant going into `depolarizing_error`.
        shots: The Number of shots to run the circuit on the back-end.
        seed: Optional seed for qiskit simulator.
    Returns:
        expval: expected values.
    """
    # initialize a qiskit noise model
    noise_model = NoiseModel()

    # we assume a depolarizing error for each gate of the standard IBM basis
    # set (u1, u2, u3)
    noise_model.add_all_qubit_quantum_error(
        depolarizing_error(noise, 1), ["u1", "u2", "u3"]
    )

    # execution of the experiment
    job = qiskit.execute(
        circuit,
        backend=QASM_SIMULATOR,
        basis_gates=["u1", "u2", "u3"],
        # we want all gates to be actually applied,
        # so we skip any circuit optimization
        optimization_level=0,
        noise_model=noise_model,
        shots=shots,
        seed_simulator=seed,
    )
    results = job.result()
    counts = results.get_counts()
    expval = counts["0"] / shots
    return expval


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
    if len(circ.clbits) > 0:
        raise ValueError(
            "This executor only works on programs with no classical bits."
        )

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
    # classical bits are included in bitstrings with a space
    # this is what breaks if you have them
    for bitstring, count in counts.items():
        expectation += eigvals[int(bitstring, 2)] * count / shots
    return expectation


def execute_with_depolarizing_noise(
    circ: QuantumCircuit, obs: np.ndarray, noise: float
) -> float:
    """Simulates the evolution of the noisy circuit and returns
    the expectation value of the observable.

    Args:
        circ: The input Qiskit circuit.
        obs: The observable to measure as a NumPy array.
        noise: The depolarizing noise strength as a float, i.e. 0.001 is 0.1%.

    Returns:
        The expectation value of obs as a float.
    """
    if len(circ.clbits) > 0:
        raise ValueError(
            "This executor only works on programs with no classical bits."
        )

    circ.snapshot("final", snapshot_type="density_matrix")

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


def execute_with_shots_and_depolarizing_noise(
    circ: QuantumCircuit, obs: np.ndarray, noise: float, shots: int,
) -> float:
    """Simulates the evolution of the noisy circuit and returns
    the expectation value of the observable.

    Args:
        circ: The input Qiskit circuit.
        obs: The observable to measure as a NumPy array.
        noise: The depolarizing noise strength as a float,
               i.e. 0.001 is 0.1%.
        shots: The number of measurements.

    Returns:
        The expectation value of obs as a float.
    """
    if len(circ.clbits) > 0:
        raise ValueError(
            "This executor only works on programs with no classical bits."
        )

    circ = copy.deepcopy(circ)
    # we need to modify the circuit to measure obs in its eigenbasis
    # we do this by appending a unitary operation
    # obtains a U s.t. obs = U diag(eigvals) U^dag
    eigvals, U = np.linalg.eigh(obs)
    circ.unitary(np.linalg.inv(U), qubits=range(circ.num_qubits))

    circ.measure_all()

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
    )
    counts = job.result().get_counts()
    expectation = 0
    # classical bits are included in bitstrings with a space
    # this is what breaks if you have them
    for bitstring, count in counts.items():
        expectation += eigvals[int(bitstring, 2)] * count / shots
    return expectation
