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
    CCNOT,
    CNOT,
    CZ,
    ISWAP,
    X,
    Y,
    Z,
    H,
    SWAP,
    Gate,
    LineQubit,
    Circuit,
    # DepolarizingChannel,
    ops,
)

from mitiq.pec.representations.biased_noise import (
    represent_operation_with_biased_noise,
)

from mitiq.pec.channels import _operation_to_choi, _circuit_to_choi


def single_qubit_biased_noise_overhead(epsilon: float, eta: float) -> float:
    """Overhead calculation similar to that presented in [Temme2017]_ and
    modified according to combined (biased) noise channel in [Strikis2021]_.

    .. [Temme2017] : Kristan Temme, Sergey Bravyi, Jay M. Gambetta,
        "Error mitigation for short-depth quantum circuits,"
        *Phys. Rev. Lett.* **119**, 180509 (2017),
        (https://arxiv.org/abs/1612.02058).

    .. [Strikis2021] : Armands Strikis, Dayue Qin, Yanzhu Chen,
        Simon C. Benjamin, and Ying Li,
        "Learning-Based Quantum Error Mitigation,"
        *PRX QUANTUM* **2**, 040330 (2021),
        (https://arxiv.org/abs/2005.07601v2).
    """
    eta1 = 1 + 3 * epsilon * (eta + 1) / (
        3 * (1 - epsilon) * (eta + 1) + epsilon * (3 * eta + 1)
    )
    eta2 = epsilon / (3 * (1 - epsilon) * (eta + 1) + epsilon * (3 * eta + 1))
    eta3 = (
        epsilon
        * (3 * eta + 1)
        / (3 * (1 - epsilon) * (eta + 1) + epsilon * (3 * eta + 1))
    )
    return eta1 + 2 * eta2 + eta3


def two_qubit_biased_noise_overhead(epsilon: float, eta: float) -> float:
    """Overhead calculation similar to that presented in [Temme2017]_ and
    modified according to combined (biased) noise channel in [Strikis2021]_.

    .. [Temme2017] : Kristan Temme, Sergey Bravyi, Jay M. Gambetta,
        "Error mitigation for short-depth quantum circuits,"
        *Phys. Rev. Lett.* **119**, 180509 (2017),
        (https://arxiv.org/abs/1612.02058).

    .. [Strikis2021] : Armands Strikis, Dayue Qin, Yanzhu Chen,
        Simon C. Benjamin, and Ying Li,
        "Learning-Based Quantum Error Mitigation,"
        *PRX QUANTUM* **2**, 040330 (2021),
        (https://arxiv.org/abs/2005.07601v2).
    """
    eta1 = 1 + 15 * epsilon * (eta + 1) / (
        15 * (1 - epsilon) * (eta + 1) + epsilon * (5 * eta + 1)
    )
    eta2 = epsilon / (15 * (1 - epsilon) * (eta + 1) + epsilon * (5 * eta + 1))
    eta3 = (
        epsilon
        * (5 * eta + 1)
        / (15 * (1 - epsilon) * (eta + 1) + epsilon * (5 * eta + 1))
    )
    return eta1 + 12 * eta2 + 3 * eta3


@pytest.mark.parametrize("epsilon", [0, 0.1, 0.7])
@pytest.mark.parametrize("eta", [0, 1, 1000])
@pytest.mark.parametrize("gate", [X, Y, Z, H])
def test_single_qubit_representation_norm(
    gate: Gate, epsilon: float, eta: float
):
    q = LineQubit(0)
    optimal_norm = single_qubit_biased_noise_overhead(epsilon, eta)
    norm = represent_operation_with_biased_noise(
        Circuit(gate(q)), epsilon, eta
    ).norm
    assert np.isclose(optimal_norm, norm)


@pytest.mark.parametrize("epsilon", [0, 0.1, 0.7])
@pytest.mark.parametrize("eta", [0, 1, 1000])
@pytest.mark.parametrize("gate", (CZ, CNOT, ISWAP, SWAP))
def test_two_qubit_representation_norm(gate: Gate, epsilon: float, eta: float):
    qreg = LineQubit.range(2)
    optimal_norm = two_qubit_biased_noise_overhead(epsilon, eta)
    norm = represent_operation_with_biased_noise(
        Circuit(gate(*qreg)), epsilon, eta
    ).norm
    assert np.isclose(optimal_norm, norm)


def test_three_qubit_biased_noise_representation_error():
    q0, q1, q2 = LineQubit.range(3)
    with pytest.raises(ValueError):
        represent_operation_with_biased_noise(
            Circuit(CCNOT(q0, q1, q2)), 0.05, 10
        )


@pytest.mark.parametrize("epsilon", [0, 0.1, 0.7])
@pytest.mark.parametrize("eta", [0, 1, 1000])
@pytest.mark.parametrize("gate", [X, Y, Z, H, CZ, CNOT, ISWAP, SWAP])
def test_biased_noise_representation_with_choi(
    gate: Gate, epsilon: float, eta: float
):
    """Tests the representation by comparing exact Choi matrices."""
    qreg = LineQubit.range(gate.num_qubits())
    ideal_choi = _operation_to_choi(gate.on(*qreg))
    op_rep = represent_operation_with_biased_noise(
        Circuit(gate.on(*qreg)), epsilon, eta
    )
    choi_components = []
    I_ = np.array([[1, 0], [0, 1]], dtype=np.complex64)
    X_ = np.array([[0, 1], [1, 0]], dtype=np.complex64)
    Y_ = np.array([[0, -1j], [1j, 0]], dtype=np.complex64)
    Z_ = np.array([[1, 0], [0, -1]], dtype=np.complex64)
    if len(qreg) == 1:
        eta1 = 1 + 3 * epsilon * (eta + 1) / (
            3 * (1 - epsilon) * (eta + 1) + epsilon * (3 * eta + 1)
        )
        eta2 = -epsilon / (
            3 * (1 - epsilon) * (eta + 1) + epsilon * (3 * eta + 1)
        )
        eta3 = (
            -epsilon
            * (3 * eta + 1)
            / (3 * (1 - epsilon) * (eta + 1) + epsilon * (3 * eta + 1))
        )
        mix = [
            (eta1, I_),
            (eta2, X_),
            (eta2, Y_),
            (eta3, Z_),
        ]
    elif len(qreg) == 2:
        eta1 = 1 + 15 * epsilon * (eta + 1) / (
            15 * (1 - epsilon) * (eta + 1) + epsilon * (5 * eta + 1)
        )
        eta2 = -epsilon / (
            15 * (1 - epsilon) * (eta + 1) + epsilon * (5 * eta + 1)
        )
        eta3 = (
            -epsilon
            * (5 * eta + 1)
            / (15 * (1 - epsilon) * (eta + 1) + epsilon * (5 * eta + 1))
        )
        mix = [
            (eta1, np.tensordot(I_, I_, axes=0)),
            (eta2, np.tensordot(I_, X_, axes=0)),
            (eta2, np.tensordot(I_, Y_, axes=0)),
            (eta3, np.tensordot(I_, Z_, axes=0)),
            (eta2, np.tensordot(X_, I_, axes=0)),
            (eta2, np.tensordot(X_, X_, axes=0)),
            (eta2, np.tensordot(X_, Y_, axes=0)),
            (eta2, np.tensordot(X_, Z_, axes=0)),
            (eta2, np.tensordot(Y_, I_, axes=0)),
            (eta2, np.tensordot(Y_, X_, axes=0)),
            (eta2, np.tensordot(Y_, Y_, axes=0)),
            (eta2, np.tensordot(Y_, Z_, axes=0)),
            (eta3, np.tensordot(Z_, I_, axes=0)),
            (eta2, np.tensordot(Z_, X_, axes=0)),
            (eta2, np.tensordot(Z_, Y_, axes=0)),
            (eta3, np.tensordot(Z_, Z_, axes=0)),
        ]

    for noisy_op, coeff in op_rep.basis_expansion.items():
        implementable_circ = noisy_op.circuit()
        # Apply noise after each sequence.
        # NOTE: noise is not applied after each operation.
        # Replace w/ custom channel:
        # biased_op = DepolarizingChannel(epsilon, len(qreg))(*qreg)
        biased_op = ops.MixedUnitaryChannel(mix)
        implementable_circ.append(biased_op)
        sequence_choi = _circuit_to_choi(implementable_circ)
        choi_components.append(coeff * sequence_choi)
    combination_choi = np.sum(choi_components, axis=0)
    assert np.allclose(ideal_choi, combination_choi, atol=10**-6)
