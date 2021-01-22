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

from typing import List

from cirq import Operation, X, Y, Z, Circuit
from mitiq import QPROGRAM
from mitiq.pec.types import OperationRepresentation, NoisyOperation
from mitiq.conversions import convert_to_mitiq, convert_from_mitiq


def depolarizing_representation(
    ideal_operation: QPROGRAM, noise_level: float
) -> OperationRepresentation:
    r"""As described in [Temme2017]_, this function maps a single-qubit
    ``ideal_operation`` :math:`\mathcal{U}_{\beta}` into its quasi-probability
    representation (QPR), which is a linear combination of noisy implementable
    operations :math:`\{\eta_{\alpha} \mathcal{O}_{\alpha}\}`.

    This function assumes depolarizing noise is the only noise present. In
    particular, it assumes that the basis of implementable operations includes
    any desired ideal operation followed by a depolarizing channel (meaning
    that all :math:`\mathcal{O}_{\alpha} = \mathcal{D} \circ \mathcal{U}`,
    where :math:`\mathcal{D}(\rho) =  (1 - \epsilon) \rho + \epsilon I/(2^n)`).
    Given that assumption, it was proven in [Takagi2020]_ that this method
    gives an optimal representation for a given depolarizing ``noise_level``
    (we can easily calculate :math:`\epsilon` from this ``noise_level`` value).
    For a single-qubit ``ideal_operation``, the optimal representation is as
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
        ideal_operation: The ideal operation (as a QPROGRAM) to represent.
        noise_level: The noise level (as a float) of the depolarizing channel.

    Returns:
        The quasi-probability representation (QPR) of the ``ideal_operation``.

    .. note::
        We allow each "noisy implementable operation" composing the
        basis of the final representation to be a sequence of 1 or 2 gates.
        This is based on the ideal assumption that one can implement *any*
        ``ideal_operation`` followed by a single depolarizing noise channel.
        When running on a simulator or QPU, this assumption breaks down for
        high ``noise_level`` values.

    .. [Temme2017] : Kristan Temme, Sergey Bravyi, Jay M. Gambetta,
        "Error mitigation for short-depth quantum circuits,"
        *Phys. Rev. Lett.* **119**, 180509 (2017),
        (https://arxiv.org/abs/1612.02058).

    .. [Takagi2020] : Ryuji Takagi,
        "Optimal resource cost for error mitigation,"
        (https://arxiv.org/abs/2006.12509).
    """
    circ, in_type = convert_to_mitiq(ideal_operation)

    post_ops: List[List[Operation]]
    qubits = circ.all_qubits()

    # The single-qubit case: linear combination of 1Q Paulis
    if len(qubits) == 1:
        q = tuple(qubits)[0]

        epsilon = 4 / 3 * noise_level
        alpha_pos = 1 + ((3 / 4) * epsilon / (1 - epsilon))
        alpha_neg = -(1 / 4) * epsilon / (1 - epsilon)

        alphas = [alpha_pos] + 3 * [alpha_neg]
        post_ops = [[]]  # for alpha_pos, we do nothing, rather than I
        post_ops += [[P(q)] for P in [X, Y, Z]]  # 1Q Paulis

    # The two-qubit case: linear combination of 2Q Paulis
    elif len(qubits) == 2:
        q0, q1 = qubits

        epsilon = 16 / 15 * noise_level
        alpha_pos = 1 + ((15 / 16) * epsilon / (1 - epsilon))
        alpha_neg = -(1 / 16) * epsilon / (1 - epsilon)

        alphas = [alpha_pos] + 15 * [alpha_neg]
        post_ops = [[]]  # for alpha_pos, we do nothing, rather than I x I
        post_ops += [[P(q0)] for P in [X, Y, Z]]  # 1Q Paulis for q0
        post_ops += [[P(q1)] for P in [X, Y, Z]]  # 1Q Paulis for q1
        post_ops += [
            [Pi(q0), Pj(q1)] for Pi in [X, Y, Z] for Pj in [X, Y, Z]
        ]  # 2Q Paulis

    else:
        raise ValueError(
            "Can only represent single- and two-qubit gates."
            "Consider pre-compiling your circuit."
        )

    # Basis of implementable operations as circuits
    imp_op_circuits = [circ + Circuit(op) for op in post_ops]

    # Convert back to input type
    imp_op_circuits = [convert_from_mitiq(c, in_type) for c in imp_op_circuits]

    # Build basis_expantion
    expansion = {NoisyOperation(c): a for c, a in zip(imp_op_circuits, alphas)}

    return OperationRepresentation(ideal_operation, expansion)
