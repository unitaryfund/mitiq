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
from mitiq.cdr._testing import random_x_z_cnot_circuit
from mitiq.pec.representations.learning import learn_biased_noise_parameters

circuit = random_x_z_cnot_circuit(
    LineQubit.range(3), n_moments=5, random_state=1
)
Rx_ops = list(circuit.findall_operations_with_gate_type(Rx))
Rz_ops = list(circuit.findall_operations_with_gate_type(Rz))


@pytest.mark.parametrize("epsilon", [0, 0.7, 1])
@pytest.mark.parametrize("eta", [0, 1, 1000])
@pytest.mark.parametrize("gate", [CNOT, Rx_ops[0][2], Rz_ops[0][2]])
def test_learn_biased_noise_parameters(epsilon, eta, gate):
    """Test the learning function with initial noise strength and noise bias
    slightly offset from the simulated noise model values"""

    observable = Observable(PauliString("XZ"), PauliString("YY"))
    # Define biased noise channel
    a = 1 - epsilon
    b = epsilon * (3 * eta + 1) / (3 * (eta + 1))
    c = epsilon / (3 * (eta + 1))

    mix = [
        (a, unitary(I)),
        (b, unitary(Z)),
        (c, unitary(X)),
        (c, unitary(Y)),
    ]

    def ideal_executor(circ: Circuit) -> np.ndarray:
        return compute_density_matrix(circ, noise_level=(0.0,))

    def noisy_executor(circ: Circuit) -> np.ndarray:
        noisy_circ = circ.with_noise(ops.MixedUnitaryChannel(mix))
        return ideal_executor(noisy_circ)

    offset = 0.01

    [epsilon_opt, eta_opt] = learn_biased_noise_parameters(
        operation=gate,
        circuit=circuit,
        ideal_executor=Executor(ideal_executor),
        noisy_executor=Executor(noisy_executor),
        num_training_circuits=10,
        epsilon0=(1 + offset) * epsilon,
        eta0=(1 + offset) * eta,
        observable=observable,
    )
    assert np.isclose(epsilon_opt, epsilon, rtol=1e-03, atol=1e-05)
    assert np.isclose(eta_opt, eta, rtol=1e-02, atol=1e-04)
