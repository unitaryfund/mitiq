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
    CZ,
    ISWAP,
    I,
    X,
    Y,
    Z,
    H,
    SWAP,
    Gate,
    LineQubit,
    Circuit,
    ops,
    unitary,
)

from mitiq.pec.representations.learning import learn_biased_noise_parameters


@pytest.mark.parametrize("epsilon", [0, 0.7, 1])
@pytest.mark.parametrize("eta", [0, 1, 1000])
@pytest.mark.parametrize("operation", [X, Y, Z, H, CZ, CNOT, ISWAP, SWAP])
@pytest.mark.parametrize("offset", [-0.1, 0.1])
def test_learn_biased_noise_parameters(epsilon, eta, operation, offset):
    a = 1 - epsilon
    b = epsilon * (3 * eta + 1) / (3 * (eta + 1))
    c = epsilon / (3 * (eta + 1))
    mix = [
        (a, unitary(I)),
        (b, unitary(Z)),
        (c, unitary(X)),
        (c, unitary(Y)),
    ]
    [epsilon_learn, eta_learn] = learn_biased_noise_parameters(
        operation=operation,
        circuit=circuit,
        ideal_executor=ideal_executor,
        noisy_executor=noisy_executor,
        num_training_circuits=10,
        epsilon0=(1 + offset) * epsilon,
        eta0=(1 + offset) * eta,
    )
    assert np.isclose(epsilon, epsilon_learn)
    assert np.isclose(eta, eta_learn)
