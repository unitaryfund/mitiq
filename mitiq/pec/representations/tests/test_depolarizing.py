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

from typing import List, Tuple
import itertools
import numpy as np
import pytest
from cirq import (
    CCNOT,
    CNOT,
    CZ,
    ISWAP,
    H,
    SWAP,
    Gate,
    LineQubit,
    Operation,
    AsymmetricDepolarizingChannel,
)

from mitiq.pec.representations.depolarizing import (
    depolarizing_decomposition,
    NON_ID_PAULIS,
)

from mitiq.pec.utils import _operation_to_choi


def my_depolarizing_channel(p: float, n_qubits: int):
    """Build a depolarizing channel from cirq.AsymmetricDepolarizingChannel
    since cirq.DepolarizingChannel is buggy when n_qubits is larger than 1."""

    # TODO: upstream bug to Cirq. Once fixed, this function can be removed.

    error_probabilities = {}
    p_depol = p / (4 ** n_qubits - 1)
    p_identity = 1.0 - p
    for pauli_tuple in itertools.product(
        ["I", "X", "Y", "Z"], repeat=n_qubits
    ):
        pauli_string = "".join(pauli_tuple)
        if pauli_string == "I" * n_qubits:
            error_probabilities[pauli_string] = p_identity
        else:
            error_probabilities[pauli_string] = p_depol
    return AsymmetricDepolarizingChannel(
        error_probabilities=error_probabilities
    )


def decomposition_overhead(
    decomposition: List[Tuple[float, List[Operation]]]
) -> float:
    """The overhead of a particular decomposition is the sum of the absolute
    values of the coefficients of the quasi-probability representation (QPR).
    """
    return float(np.sum(np.abs([a for a, _ in decomposition])))


def single_qubit_depolarizing_overhead(noise_level: float) -> float:
    """See [Temme2017]_ for more information.

    .. [Temme2017] : Kristan Temme, Sergey Bravyi, Jay M. Gambetta,
        "Error mitigation for short-depth quantum circuits,"
        *Phys. Rev. Lett.* **119**, 180509 (2017),
        (https://arxiv.org/abs/1612.02058).
    """
    epsilon = 4 / 3 * noise_level
    return (1 + epsilon / 2) / (1 - epsilon)


def two_qubit_depolarizing_overhead(noise_level: float) -> float:
    """See [Temme2017]_ for more information.

        .. [Temme2017] : Kristan Temme, Sergey Bravyi, Jay M. Gambetta,
            "Error mitigation for short-depth quantum circuits,"
            *Phys. Rev. Lett.* **119**, 180509 (2017),
            (https://arxiv.org/abs/1612.02058).
    """
    epsilon = 16 / 15 * noise_level
    return (1 + 7 * epsilon / 8) / (1 - epsilon)


def test_single_qubit_depolarizing_decomposition():
    q = LineQubit(0)
    noise_level = 0.05
    optimal_overhead = single_qubit_depolarizing_overhead(noise_level)
    assert all(
        np.isclose(
            optimal_overhead,
            decomposition_overhead(
                depolarizing_decomposition(P(q), noise_level)
            ),
        )
        for P in NON_ID_PAULIS
    )


def test_two_qubit_depolarizing_decomposition():
    q0, q1 = LineQubit.range(2)
    noise_level = 0.05
    optimal_overhead = two_qubit_depolarizing_overhead(noise_level)
    assert all(
        np.isclose(
            optimal_overhead,
            decomposition_overhead(
                depolarizing_decomposition(G(q0, q1), noise_level)
            ),
        )
        for G in (CZ, CNOT, ISWAP, SWAP)
    )


def test_three_qubit_depolarizing_decomposition():
    q0, q1, q2 = LineQubit.range(3)
    noise_level = 0.05
    with pytest.raises(ValueError):
        depolarizing_decomposition(CCNOT(q0, q1, q2), noise_level)


@pytest.mark.parametrize("noise", [0, 0.1, 0.5, 1.0])
@pytest.mark.parametrize("gate", NON_ID_PAULIS + [H, CZ, CNOT, ISWAP, SWAP])
def test_depolarizing_decomposition_with_Choi(gate: Gate, noise: float):
    """Tests the decomposition by comparing exact Choi matrices."""
    qreg = LineQubit.range(gate.num_qubits())
    ideal_choi = _operation_to_choi(gate.on(*qreg))
    op_decomp = depolarizing_decomposition(gate.on(*qreg), noise)
    choi_components = []
    for coeff, imp_seq in op_decomp:
        # Apply noise after each sequence.
        # NOTE: noise is not applied after each operation.
        depolarizing_op = my_depolarizing_channel(noise, len(qreg))(*qreg)
        noisy_sequence = [imp_seq] + [depolarizing_op]
        sequence_choi = _operation_to_choi(noisy_sequence)
        choi_components.append(coeff * sequence_choi)
    combination_choi = np.sum(choi_components, axis=0)
    assert np.allclose(ideal_choi, combination_choi, atol=10 ** -6)
