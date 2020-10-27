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

from cirq import Operation, X, Y, Z

NON_ID_PAULIS = [X, Y, Z]


def depolarizing_decomposition(
    ideal_operation: Operation, noise_level: float
) -> List[Tuple[float, List[Operation]]]:
    r"""As described in [Temme2017]_, optimally decompose a single-qubit
    ``ideal_operation`` :math:`\mathcal{U}_{\beta}` into its quasi-probability
    representation (QPR), which is a linear combination of noisy implementable
    operations :math:`\{\eta_{\alpha} \mathcal{O}_{\alpha}\}`.

    This function assumes depolarizing noise is the only noise present. In
    particular, it assumes that the basis of implementable operations includes
    any desired ideal operation followed by a depolarizing channel (meaning
    that all :math:`\mathcal{O}_{\alpha} = \mathcal{D} \circ \mathcal{U}`,
    where :math:`\mathcal{D}(\rho) =  (1 - \epsilon) \rho + \epsilon I/(2^n)`).
    Given that assumption, it was proven in [Takagi2020]_ that this method
    gives an optimal decomposition for a given depolarizing ``noise_level``
    (we can easily calculate :math:`\epsilon` from this ``noise_level`` value).
    For a single-qubit ``ideal_operation``, the optimal decomposition is as
    follows:

    .. math::
         \mathcal{U}_{\beta} = \eta_1 \mathcal{O}_1 + \eta_2 \mathcal{O}_2 +
                               \eta_3 \mathcal{O}_3 + \eta_4 \mathcal{O}_4

    .. math::
        \eta_1 =1 + \frac{3}{4} \frac{\epsilon}{1- \epsilon},
        \qquad \mathcal{O}_1 = \mathcal{D} \circ \mathcal{I} \circ \mathcal{U}

        \eta_2 =- \frac{1}{4}\frac{\epsilon}{1- \epsilon} , \qquad
        \mathcal{O}_2 = \mathcal{D} \circ \mathcal{X} \circ \mathcal{U}

        \eta_3 =- \frac{1}{4}\frac{\epsilon}{1- \epsilon} , \qquad
        \mathcal{O}_3 = \mathcal{D} \circ \mathcal{Y} \circ \mathcal{U}

        \eta_4 =- \frac{1}{4}\frac{\epsilon}{1- \epsilon} , \qquad
        \mathcal{O}_4 = \mathcal{D} \circ \mathcal{Z} \circ \mathcal{U}

    Args:
        ideal_operation: The input ideal operation (gate + qubit) to decompose.
        noise_level: The noise level (as a float) of the depolarizing channel.

    Returns:
        The quasi-probability representation (QPR) of the ``ideal_operation``,
        encoded as a list of tuples, where the first element in each tuple is
        a float coefficient, and the second element is a list of Cirq
        ``Operation`` objects to replace the ``ideal_operation`` with.

    .. note::
        In the description we say "noisy implementable operation" but we return
        lists of operations. This is a subtle point, relying on our assumption
        that we can implement *any* ``ideal_operation`` followed by a single
        depolarizing noise channel. When running on a simulator or QPU, this
        assumption breaks down for high ``noise_level`` values.

    .. [Temme2017] : Kristan Temme, Sergey Bravyi, Jay M. Gambetta,
        "Error mitigation for short-depth quantum circuits,"
        *Phys. Rev. Lett.* **119**, 180509 (2017),
        (https://arxiv.org/abs/1612.02058).

    .. [Takagi2020] : Ryuji Takagi,
        "Optimal resource cost for error mitigation,"
        (https://arxiv.org/abs/2006.12509).
    """
    post_ops: List[List[Operation]]
    qubits = ideal_operation.qubits

    # the single-qubit case: linear combination of 1Q Paulis
    if len(qubits) == 1:
        q = ideal_operation.qubits[0]

        epsilon = 4 / 3 * noise_level
        alpha_pos = 1 + ((3 / 4) * epsilon / (1 - epsilon))
        alpha_neg = -(1 / 4) * epsilon / (1 - epsilon)

        alphas = [alpha_pos] + 3 * [alpha_neg]
        post_ops = [[]]  # for alpha_pos, we do nothing, rather than I
        post_ops += [[P(q)] for P in NON_ID_PAULIS]  # 1Q Paulis

    # the two-qubit case: linear combination of 2Q Paulis
    elif len(qubits) == 2:
        q0, q1 = qubits

        epsilon = 16 / 15 * noise_level
        alpha_pos = 1 + ((15 / 16) * epsilon / (1 - epsilon))
        alpha_neg = -(1 / 16) * epsilon / (1 - epsilon)

        alphas = [alpha_pos] + 15 * [alpha_neg]
        post_ops = [[]]  # for alpha_pos, we do nothing, rather than I x I
        post_ops += [[P(q0)] for P in NON_ID_PAULIS]  # 1Q Paulis for q0
        post_ops += [[P(q1)] for P in NON_ID_PAULIS]  # 1Q Paulis for q1
        post_ops += [
            [Pi(q0), Pj(q1)] for Pi in NON_ID_PAULIS for Pj in NON_ID_PAULIS
        ]  # 2Q Paulis

    else:
        raise ValueError(
            "Can only decompose single- and two-qubit gates."
            "Consider pre-compiling your circuit."
        )

    return [(a, [ideal_operation] + p) for a, p in zip(alphas, post_ops)]
