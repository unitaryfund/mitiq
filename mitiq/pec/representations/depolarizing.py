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
from itertools import product

from cirq import Operation, X, Y, Z, Circuit
from mitiq import QPROGRAM
from mitiq.pec.types import OperationRepresentation, NoisyOperation
from mitiq.conversions import convert_to_mitiq, convert_from_mitiq


def represent_operation_with_global_depolarizing_noise(
    ideal_operation: QPROGRAM, noise_level: float
) -> OperationRepresentation:
    r"""As described in [Temme2017]_, this function maps an
    ``ideal_operation`` :math:`\mathcal{U}` into its quasi-probability
    representation, which is a linear combination of noisy implementable
    operations :math:`\sum_\alpha \eta_{\alpha} \mathcal{O}_{\alpha}`.

    This function assumes a depolarizing noise model, and more precicely,
    that the following noisy operations are implementable
    :math:`\mathcal{O}_{\alpha} = \mathcal{D} \circ \mathcal P_\alpha
    \circ \mathcal{U}`, where :math:`\mathcal{U}` is the unitary associated
    to the input ``ideal_operation`` acting on :math:`k` qubits,
    :math:`\mathcal{P}_\alpha` is a Pauli operation and
    :math:`\mathcal{D}(\rho) = (1 - \epsilon) \rho + \epsilon I/2^k` is a
    depolarizing channel (:math:`\epsilon` is a simple function of
    ``noise_level``).

    For a single-qubit ``ideal_operation``, the representation is as
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

    It was proven in [Takagi2020]_ that, under suitable assumptions,
    this representation is optimal (minimum 1-norm).

    Args:
        ideal_operation: The ideal operation (as a QPROGRAM) to represent.
        noise_level: The noise level (as a float) of the depolarizing channel.

    Returns:
        The quasi-probability representation of the ``ideal_operation``.

    .. note::
        This representation is based on the ideal assumption that one
        can append Pauli gates to a noisy operation without introducing
        additional noise. For a beckend which violates this assumption,
        it remains a good approximation for small values of ``noise_level``.

    .. note::
        The input ``ideal_operation`` is typically a QPROGRAM with a single
        gate but could also correspond to a sequence of more gates.
        This is possible as long as the unitary associated to the input
        QPROGRAM, followed by a single final depolarizing channel, is
        physically implementable.
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


def represent_operation_with_local_depolarizing_noise(
    ideal_operation: QPROGRAM, noise_level: float
) -> OperationRepresentation:
    r"""As described in [Temme2017]_, this function maps an
    ``ideal_operation`` :math:`\mathcal{U}` into its quasi-probability
    representation, which is a linear combination of noisy implementable
    operations :math:`\sum_\alpha \eta_{\alpha} \mathcal{O}_{\alpha}`.

    This function assumes a (local) single-qubit depolarizing noise model even
    for multi-qubit operations. More precicely, it assumes that the following
    noisy operations are implementable :math:`\mathcal{O}_{\alpha} =
    \mathcal{D}^{\otimes k} \circ \mathcal P_\alpha \circ \mathcal{U}`,
    where :math:`\mathcal{U}` is the unitary associated
    to the input ``ideal_operation`` acting on :math:`k` qubits,
    :math:`\mathcal{P}_\alpha` is a Pauli operation and
    :math:`\mathcal{D}(\rho) = (1 - \epsilon) \rho + \epsilon I/2` is a
    single-qubit depolarizing channel (:math:`\epsilon` is a simple function
    of ``noise_level``).

    More information about the quasi-probability representation of depolarizing
    noise can be found in the docstring of
    ``represent_operation_with_global_depolarizing_noise``.

    Args:
        ideal_operation: The ideal operation (as a QPROGRAM) to represent.
        noise_level: The noise level of each depolarizing channel.

    Returns:
        The quasi-probability representation of the ``ideal_operation``.

    .. note::
        The input ``ideal_operation`` is typically a QPROGRAM with a single
        gate but could also correspond to a sequence of more gates.
        This is possible as long as the unitary associated to the input
        QPROGRAM, followed by a single final depolarizing channel, is
        physically implementable.

    .. [Temme2017] : Kristan Temme, Sergey Bravyi, Jay M. Gambetta,
        "Error mitigation for short-depth quantum circuits,"
        *Phys. Rev. Lett.* **119**, 180509 (2017),
        (https://arxiv.org/abs/1612.02058).
    """
    circ, in_type = convert_to_mitiq(ideal_operation)

    qubits = circ.all_qubits()

    if len(qubits) == 1:
        return represent_operation_with_global_depolarizing_noise(
            ideal_operation, noise_level,
        )

    # The two-qubit case: tensor product of two depolarizing channels.
    elif len(qubits) == 2:
        q0, q1 = qubits

        # Single-qubit representation coefficients.
        epsilon = noise_level * 4 / 3
        c_neg = -(1 / 4) * epsilon / (1 - epsilon)
        c_pos = 1 - 3 * c_neg

        imp_op_circuits = []
        alphas = []

        # The zero-pauli term in the linear combination
        imp_op_circuits.append(circ)
        alphas.append(c_pos * c_pos)

        # The single-pauli terms in the linear combination
        for qubit in qubits:
            for pauli in [X, Y, Z]:
                imp_op_circuits.append(circ + Circuit(pauli(qubit)))
                alphas.append(c_neg * c_pos)

        # The two-pauli terms in the linear combination
        for pauli_0, pauli_1 in product([X, Y, Z], repeat=2):
            imp_op_circuits.append(circ + Circuit(pauli_0(q0), pauli_1(q1)))
            alphas.append(c_neg * c_neg)

    else:
        raise ValueError(
            "Can only represent single- and two-qubit gates."
            "Consider pre-compiling your circuit."
        )

    # Convert back to input type
    imp_op_circuits = [convert_from_mitiq(c, in_type) for c in imp_op_circuits]

    # Build basis_expantion
    expansion = {NoisyOperation(c): a for c, a in zip(imp_op_circuits, alphas)}

    return OperationRepresentation(ideal_operation, expansion)
