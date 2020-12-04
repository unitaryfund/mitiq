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

import numpy as np
import pytest
from cirq import Circuit, CCNOT, CNOT, CZ, ISWAP, LineQubit, SWAP

from mitiq.pec.types import OperationDecomposition
from mitiq.pec.decomposition.depolarizing import (
    depolarizing_decomposition,
    NON_ID_PAULIS,
)


def decomposition_overhead(decomposition: OperationDecomposition) -> float:
    """The overhead of a particular decomposition is the sum of the absolute
    values of the coefficients of the quasi-probability representation (QPR).
    """
    return decomposition.norm


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
                depolarizing_decomposition(Circuit(P(q)), noise_level)
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
                depolarizing_decomposition(Circuit(G(q0, q1)), noise_level)
            ),
        )
        for G in (CZ, CNOT, ISWAP, SWAP)
    )


def test_three_qubit_depolarizing_decomposition():
    q0, q1, q2 = LineQubit.range(3)
    noise_level = 0.05
    with pytest.raises(ValueError):
        depolarizing_decomposition(Circuit(CCNOT(q0, q1, q2)), noise_level)
