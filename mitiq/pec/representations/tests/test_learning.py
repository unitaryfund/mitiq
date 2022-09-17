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
import qiskit
from mitiq import Executor, Observable, PauliString
from mitiq.interface.mitiq_qiskit import qiskit_utils
from mitiq.interface.mitiq_qiskit.conversions import to_qiskit
from mitiq.interface.mitiq_cirq import compute_density_matrix
from mitiq.cdr import generate_training_circuits
from mitiq.cdr._testing import random_x_z_cnot_circuit
from mitiq.pec.representations.learning import (
    biased_noise_loss_function,
    learn_noise_parameters_from_unmitigated_data,
)

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
    loss = biased_noise_loss_function(
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
    loss = biased_noise_loss_function(
        params=[0, 0],
        operations_to_mitigate=operations,
        training_circuits=training_circuits,
        ideal_values=ideal_values,
        noisy_executor=noisy_executor,
        pec_kwargs=pec_kwargs,
        observable=observable,
    )
    assert np.isclose(loss, 0)


offset = 0.01


@pytest.mark.parametrize("epsilon", [0, 0.05, 0.1])
def test_learn_noise_parameters_from_unmitigated_data(epsilon):
    """Test the learning function with initial noise strength with a small
    offset from the simulated noise model values"""

    def noisy_execute(circ: Circuit) -> np.ndarray:
        noisy_circ = circ.with_noise(
            biased_noise_channel(epsilon=epsilon, eta=0)
        )
        return ideal_execute(noisy_circ)

    noisy_executor = Executor(noisy_execute)

    if epsilon == 0:
        epsilon0 = offset
    else:
        epsilon0 = (1 + offset) * epsilon

    [success, epsilon_opt] = learn_noise_parameters_from_unmitigated_data(
        circuit=circuit,
        epsilon0=epsilon0,
        noisy_executor=noisy_executor,
        num_training_circuits=5,
        fraction_non_clifford=0.2,
        training_random_state=np.random.RandomState(1),
        observable=observable,
    )
    assert success
    assert abs(epsilon_opt - epsilon) < abs(epsilon0 - epsilon)


def test_learn_noise_parameters_unmitigated_qiskit():
    """Test the learning function with initial noise strength
    with a small offset from the simulated noise model values"""
    epsilon = 0.05

    def noisy_execute_qiskit(circ: qiskit.QuantumCircuit) -> float:
        noise_model = qiskit_utils.initialized_depolarizing_noise(epsilon)
        return qiskit_utils.execute_with_noise(
            circ, observable.matrix(), noise_model
        )

    noisy_executor_qiskit = Executor(noisy_execute_qiskit)
    epsilon0 = (1 + offset) * epsilon

    qiskit_circuit = to_qiskit(circuit)
    [success, _] = learn_noise_parameters_from_unmitigated_data(
        circuit=qiskit_circuit,
        epsilon0=epsilon0,
        noisy_executor=noisy_executor_qiskit,
        num_training_circuits=5,
        fraction_non_clifford=0.2,
        training_random_state=np.random.RandomState(1),
        observable=observable,
    )
    assert success
