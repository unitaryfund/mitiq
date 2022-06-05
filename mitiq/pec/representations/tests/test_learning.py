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
    CNOT,
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
from mitiq.pec.representations.learning import biased_noise_loss_function

circuit = random_x_z_cnot_circuit(
    LineQubit.range(2), n_moments=5, random_state=1
)

observable = Observable(PauliString("XZ"), PauliString("YY"))

training_circuits = generate_training_circuits(
    circuit=circuit,
    num_training_circuits=3,
    fraction_non_clifford=0,
    method_select="uniform",
    method_replace="closest",
)

Rx_ops = list(circuit.findall_operations_with_gate_type(Rx))
Rz_ops = list(circuit.findall_operations_with_gate_type(Rz))


def ideal_execute(circ: Circuit) -> np.ndarray:
    return compute_density_matrix(circ, noise_level=(0.0,))


ideal_executor = Executor(ideal_execute)
ideal_values = np.array(
    [ideal_executor.evaluate(t, observable) for t in training_circuits]
)


def biased_noise_channel(epsilon, eta):
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
@pytest.mark.parametrize("gate", [CNOT, Rx_ops[0][2]])
def test_biased_noise_loss_function(epsilon, eta, gate):
    """Test that the biased noise loss function value (calculated with error
    mitigation) is smaller than the loss calculated with the noisy
    (unmitigated) executor"""

    def noisy_execute(circ: Circuit) -> np.ndarray:
        noisy_circ = circ.with_noise(biased_noise_channel(epsilon, eta))
        return ideal_execute(noisy_circ)

    noisy_executor = Executor(noisy_execute)
    noisy_values = np.array(
        [noisy_executor.evaluate(t, observable) for t in training_circuits]
    )
    loss = biased_noise_loss_function(
        [epsilon, eta],
        gate,
        training_circuits,
        ideal_values,
        noisy_executor,
        observable,
    )

    assert loss <= np.sum((noisy_values - ideal_values) ** 2) / len(
        training_circuits
    )


@pytest.mark.parametrize("gate", [CNOT, Rx_ops[0][2], Rz_ops[0][2]])
def test_biased_noise_compare_ideal(gate):
    def noisy_execute(circ: Circuit) -> np.ndarray:
        noisy_circ = circ.with_noise(biased_noise_channel(0, 0))
        return ideal_execute(noisy_circ)

    noisy_executor = Executor(noisy_execute)
    loss = biased_noise_loss_function(
        [0, 0],
        gate,
        training_circuits,
        ideal_values,
        noisy_executor,
        observable,
    )
    assert np.isclose(loss, 0)
