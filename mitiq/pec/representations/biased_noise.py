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
"""Function to generate representations with biased noise."""

from typing import List

from cirq import (
    Operation,
    X,
    Y,
    Z,
    Circuit,
)

from mitiq import QPROGRAM
from mitiq.pec import OperationRepresentation, NoisyOperation
from mitiq.interface import convert_to_mitiq, convert_from_mitiq


def represent_operation_with_biased_noise(
    ideal_operation: QPROGRAM,
    epsilon: float,
    eta: float,
) -> OperationRepresentation:
    r"""As described in [Strikis2021]_, this function maps an
    ``ideal_operation`` :math:`\mathcal{U}` into its quasi-probability
    representation, which is a linear combination of noisy implementable
    operations :math:`\sum_\alpha \eta_{\alpha} \mathcal{O}_{\alpha}`.

    This function assumes a combined depolarizing and dephasing noise model
    with a bias factor :math:`\eta` and that the following noisy
    operations are implementable
    :math:`\mathcal{O}_{\alpha} = \mathcal{D} \circ \mathcal P_\alpha
    \circ \mathcal{U}`, where :math:`\mathcal{U}` is the unitary associated
    to the input ``ideal_operation`` acting on :math:`k` qubits,
    :math:`\mathcal{P}_\alpha` is a Pauli operation and
    :math:`\mathcal{D}(\rho) = (1 - \epsilon) \rho + \epsilon I/2^k` is a
    combined (biased) depolarizing and dephasing channel.

    For a single-qubit ``ideal_operation``, the representation is as
    follows:

    .. math::
         \mathcal{U}_{\beta} = \eta_1 \mathcal{O}_1 + \eta_2 \mathcal{O}_2 +
                               \eta_3 \mathcal{O}_3 + \eta_4 \mathcal{O}_4

    .. math::
        \eta_1 = 1 + \frac{3 \epsilon (\eta + 1)}{3 (1 - \epsilon)(\eta + 1) +
                 \epsilon (3 \eta + 1)} ,
        \qquad \mathcal{O}_1 = \mathcal{D} \circ \mathcal{I} \circ \mathcal{U}

        \eta_2 = - \frac{\epsilon}{3 (1 - \epsilon)(\eta + 1) +
                 \epsilon (3 \eta + 1)} ,
        \qquad \mathcal{O}_2 = \mathcal{D} \circ \mathcal{X} \circ \mathcal{U}

        \eta_3 = - \frac{\epsilon}{3 (1 - \epsilon)(\eta + 1) +
                 \epsilon (3 \eta + 1)} ,
        \qquad \mathcal{O}_3 = \mathcal{D} \circ \mathcal{Y} \circ \mathcal{U}

        \eta_4 = - \frac{\epsilon (\eta + 1)}{3 (1 - \epsilon)(\eta + 1) +
                 \epsilon (3 \eta + 1)} ,
        \qquad \mathcal{O}_4 = \mathcal{D} \circ \mathcal{Z} \circ \mathcal{U}

    Args:
        ideal_operation: The ideal operation (as a QPROGRAM) to represent.
        epsilon: The local noise severity (as a float) of the combined channel.
        eta: The noise bias between combined dephasing and depolarizing
        channelswith :math:`\eta = 0` describing a fully depolarizing channel
        and :math:`\eta = inf` describing a fully dephasing channel.

    Returns:
        The quasi-probability representation of the ``ideal_operation``.

    .. note::
        This representation is based on the ideal assumption that one
        can append Pauli gates to a noisy operation without introducing
        additional noise. For a backend which violates this assumption,
        it remains a good approximation for small values of ``noise_level``.

    .. note::
        The input ``ideal_operation`` is typically a QPROGRAM with a single
        gate but could also correspond to a sequence of more gates.
        This is possible as long as the unitary associated to the input
        QPROGRAM, followed by a single final biased noise channel, is
        physically implementable.
    """
    circ, in_type = convert_to_mitiq(ideal_operation)

    post_ops: List[List[Operation]]
    qubits = circ.all_qubits()

    if len(qubits) == 1:
        q = tuple(qubits)[0]

        a = 1 - epsilon
        b = epsilon * (3 * eta + 1) / (3 * (eta + 1))
        c = epsilon / (3 * (eta + 1))
        d = 1 / (((a - b) ** 2) - 4 * c**2)
        alpha_1 = (a * (a - b) - 2 * c**2) * d / (a + b)
        alpha_2 = -c * d
        alpha_3 = 2 * c * d

        alphas = [alpha_1] + 2 * [alpha_2] + [alpha_3]
        post_ops = [[]]  # for eta_1, we do nothing, rather than I
        post_ops += [[P(q)] for P in [X, Y, Z]]  # 1Q Paulis

        # The two-qubit case: linear combination of 2Q Paulis
    elif len(qubits) == 2:
        q0, q1 = qubits

        alpha_1 = 1 + 15 * epsilon * (eta + 1) / (
            15 * (1 - epsilon) * (eta + 1) + epsilon * (5 * eta + 1)
        )
        alpha_2 = -epsilon / (
            15 * (1 - epsilon) * (eta + 1) + epsilon * (5 * eta + 1)
        )
        alpha_3 = (
            -epsilon
            * (5 * eta + 1)
            / (15 * (1 - epsilon) * (eta + 1) + epsilon * (5 * eta + 1))
        )

        alphas = [alpha_1] + 12 * [alpha_2] + 3 * [alpha_3]
        post_ops = [[]]  # for eta_1, we do nothing, rather than I x I
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
    # Basis of implementable operations as circuits.
    imp_op_circuits = [circ + Circuit(op) for op in post_ops]

    # Convert back to input type.
    imp_op_circuits = [convert_from_mitiq(c, in_type) for c in imp_op_circuits]

    # Build basis expansion.
    expansion = {NoisyOperation(c): a for c, a in zip(imp_op_circuits, alphas)}
    return OperationRepresentation(ideal_operation, expansion)
