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
    DepolarizingChannel,
)


from mitiq.pec.representations.biased_noise import (
    represent_operation_with_biased_noise,
)

from mitiq.pec.channels import _operation_to_choi, _circuit_to_choi
from mitiq.utils import _equal
from mitiq.interface import convert_to_mitiq, convert_from_mitiq


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


@pytest.mark.parametrize("circuit_type", ["cirq", "qiskit", "pyquil"])
def test_represent_operations_with_biased_noise(circuit_type: str):
    """Tests all operation representations are created."""
    qreg = LineQubit.range(2)
    circ_mitiq = Circuit([CNOT(*qreg), H(qreg[0]), Y(qreg[1]), CNOT(*qreg)])
    circ = convert_from_mitiq(circ_mitiq, circuit_type)

    reps = represent_operation_with_biased_noise(
        ideal_operation=circ, epsilon=0.1, eta=1
    )

    # For each operation in circ we should find its representation
    for op in convert_to_mitiq(circ)[0].all_operations():
        found = False
        for rep in reps:
            if _equal(rep.ideal, Circuit(op), require_qubit_equality=True):
                found = True
        assert found

    # The number of reps. should match the number of unique operations
    assert len(reps) == 3


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
    for noisy_op, coeff in op_rep.basis_expansion.items():
        implementable_circ = noisy_op.circuit()
        # Apply noise after each sequence.
        # NOTE: noise is not applied after each operation.
        # Replace w/ custom channel:
        biased_op = DepolarizingChannel(epsilon, len(qreg))(*qreg)
        implementable_circ.append(biased_op)
        sequence_choi = _circuit_to_choi(implementable_circ)
        choi_components.append(coeff * sequence_choi)
    combination_choi = np.sum(choi_components, axis=0)
    assert np.allclose(ideal_choi, combination_choi, atol=10**-6)
