# Copyright (C) 2022 Unitary Fund
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

import numpy as np
import pytest
from cirq import (
    CXPowGate,
    MixedUnitaryChannel,
    Rx,
    Rz,
    I,
    X,
    Y,
    Z,
    LineQubit,
    Circuit,
    ops,
    unitary,
)
from mitiq import Executor, Observable, PauliString
from mitiq.interface.mitiq_cirq import compute_density_matrix
from mitiq.cdr import generate_training_circuits
from mitiq.cdr._testing import random_x_z_cnot_circuit
from mitiq.pec.representations.learning import _biased_noise_loss_function

circuit = random_x_z_cnot_circuit(
    LineQubit.range(2), n_moments=5, random_state=1
)

# Set number of samples used to calculate mitigated value in loss function
pec_kwargs = {"num_samples": 100}

observable = Observable(PauliString("XZ"), PauliString("YY"))

training_circuits = generate_training_circuits(
    circuit=circuit,
    num_training_circuits=3,
    fraction_non_clifford=0,
    method_select="uniform",
    method_replace="closest",
)

CNOT_ops = list(circuit.findall_operations_with_gate_type(CXPowGate))
Rx_ops = list(circuit.findall_operations_with_gate_type(Rx))
Rz_ops = list(circuit.findall_operations_with_gate_type(Rz))


def ideal_execute(circ: Circuit) -> np.ndarray:
    return compute_density_matrix(circ, noise_level=(0.0,))


ideal_executor = Executor(ideal_execute)
ideal_values = np.array(
    [ideal_executor.evaluate(t, observable) for t in training_circuits]
)


def biased_noise_channel(epsilon: float, eta: float) -> MixedUnitaryChannel:
    a = 1 - epsilon
    b = epsilon * (3 * eta + 1) / (3 * (eta + 1))
    c = epsilon / (3 * (eta + 1))

    mix = [
        (a, unitary(I)),
        (b, unitary(Z)),
        (c, unitary(X)),
        (c, unitary(Y)),
    ]
    return ops.MixedUnitaryChannel(mix)


@pytest.mark.parametrize("epsilon", [0, 0.7, 1])
@pytest.mark.parametrize("eta", [0, 1])
@pytest.mark.parametrize(
    "operations", [[Circuit(CNOT_ops[0][1])], [Circuit(Rx_ops[0][1])]]
)
def test_biased_noise_loss_function(epsilon, eta, operations):
    """Test that the biased noise loss function value (calculated with error
    mitigation) is less than (or equal to) the loss calculated with the noisy
    (unmitigated) executor"""

    def noisy_execute(circ: Circuit) -> np.ndarray:
        noisy_circ = circ.with_noise(biased_noise_channel(epsilon, eta))
        return ideal_execute(noisy_circ)

    noisy_executor = Executor(noisy_execute)
    noisy_values = np.array(
        [noisy_executor.evaluate(t, observable) for t in training_circuits]
    )
    loss = _biased_noise_loss_function(
        params=[epsilon, eta],
        operations_to_mitigate=operations,
        training_circuits=training_circuits,
        ideal_values=ideal_values,
        noisy_executor=noisy_executor,
        pec_kwargs=pec_kwargs,
        observable=observable,
    )

    assert loss <= np.sum((noisy_values - ideal_values) ** 2) / len(
        training_circuits
    )


@pytest.mark.parametrize(
    "operations", [[Circuit(CNOT_ops[0][1])], [Circuit(Rz_ops[0][1])]]
)
def test_biased_noise_loss_compare_ideal(operations):
    """Test that the loss function is zero when the noise strength is zero"""

    def noisy_execute(circ: Circuit) -> np.ndarray:
        noisy_circ = circ.with_noise(biased_noise_channel(0, 0))
        return ideal_execute(noisy_circ)

    noisy_executor = Executor(noisy_execute)
    loss = _biased_noise_loss_function(
        params=[0, 0],
        operations_to_mitigate=operations,
        training_circuits=training_circuits,
        ideal_values=ideal_values,
        noisy_executor=noisy_executor,
        pec_kwargs=pec_kwargs,
        observable=observable,
    )
    assert np.isclose(loss, 0)

  
import qiskit
from qiskit import QuantumCircuit
from qiskit.providers.aer.noise import NoiseModel
from mitiq.interface.mitiq_qiskit.qiskit_utils import (
    initialized_depolarizing_noise,
)
def test_loss_function_qiskit():


    def noisy_execute_qiskit(
        circuit: QuantumCircuit,
        obs: np.ndarray,
        noise_model: NoiseModel,
        shots: int,
    ) -> float:
        """Simulates the evolution of the noisy circuit and returns
        the statistical estimate of the expectation value of the observable.
        """
        # Avoid mutating circuit
        circ = circuit.copy()
        # we need to modify the circuit to measure obs in its eigenbasis
     # we do this by appending a unitary operation
        # obtains a U s.t. obs = U diag(eigvals) U^dag
        eigvals, U = np.linalg.eigh(obs)
        circ.unitary(np.linalg.inv(U), qubits=range(circ.num_qubits))

        circ.measure_all()

        if noise_model is None:
            basis_gates = None
        else:
            basis_gates = noise_model.basis_gates

    # execution of the experiment
    job = qiskit.execute(
        circ,
        backend=qiskit.Aer.get_backend("aer_simulator"),
        backend_options={"method": "density_matrix"},
        noise_model=noise_model,
        basis_gates=basis_gates,
        shots=shots,
    )
    counts = job.result().get_counts()
    expectation = 0

    for bitstring, count in counts.items():
        expectation += (
            eigvals[int(bitstring[0 : circ.num_qubits], 2)] * count / shots
        )
    return expectation


noisy_executor = noisy_execute_qiskit(circuit=circuit, obs=obsevable, noise_model=initialized_depolarizing_noise(NOISE))
