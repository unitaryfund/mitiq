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


def represent_operation_with_local_biased_noise(
    ideal_operation: QPROGRAM,
    epsilon: float,
    eta: float,
) -> OperationRepresentation:
    r"""This function maps an
    ``ideal_operation`` :math:`\mathcal{U}` into its quasi-probability
    representation, which is a linear combination of noisy implementable
    operations :math:`\sum_\alpha \eta_{\alpha} \mathcal{O}_{\alpha}`.

    This function assumes a combined depolarizing and dephasing noise model
    with a bias factor :math:`\eta` (see :cite:`Strikis_2021_PRXQuantum`)
    and that the following noisy operations are implementable,
    where :math:`\mathcal{U}` is the unitary associated
    to the input ``ideal_operation`` acting on :math:`k` qubits,
    :math:`\mathcal{P}_\alpha` is a Pauli operation and
    ..math::`\mathcal{D}(\epsilon) = (1 - \epsilon)[[mathbb{1}] +
    \epsilon(frac{\eta}{\eta + 1}mathcal{D^{*}_ph}
    + frac{1}{eta + 1}mathcal{D^{*}_pol})`is a combined (biased) depolarizing
    and dephasing channel, with :math:`\mathcal{D}^{*}_ph
    = frac{1}{3}\sum\_{\mu\in\{I,Z\}^{\otimes2}\setminusI^{\otimes}2}}[\mu]`
    and :math:`\mathcal{D}^{*}_pol
    = frac{1}{15}\sum\_{
        \mu\in\{I,X,Y,Z\}^{\otimes2}\setminusI^{\otimes2}}[\mu]`

    Args:
        ideal_operation: The ideal operation (as a QPROGRAM) to represent.
        epsilon: The local noise severity (as a float) of the combined channel.
        eta: The noise bias between combined dephasing and depolarizing
        channelswith :math:`\eta = 0` describing a fully depolarizing channel
        and :math:`\eta = \inf` describing a fully dephasing channel.

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

    # Calculate coefficients in basis expansion in terms of eta and epsilon
    a = 1 - epsilon
    b = epsilon * (3 * eta + 1) / (3 * (eta + 1))
    c = epsilon / (3 * (eta + 1))
    alpha = (a**2 + a * b - 2 * c**2) / (
        a**3
        + a**2 * b
        - a * b**2
        - 4 * a * c**2
        - b**3
        + 4 * b * c**2
    )
    beta = (-a * b - b**2 + 2 * c**2) / (
        a**3
        + a**2 * b
        - a * b**2
        - 4 * a * c**2
        - b**3
        + 4 * b * c**2
    )
    gamma = -c / (a**2 + 2 * a * b + b**2 - 4 * c**2)
    if len(qubits) == 1:
        q = tuple(qubits)[0]
        alphas = [alpha] + 2 * [gamma] + [beta]
        post_ops = [[]]  # for eta_1, we do nothing, rather than I
        post_ops += [[P(q)] for P in [X, Y, Z]]  # 1Q Paulis

        # The two-qubit case: linear combination of 2Q Paulis
    elif len(qubits) == 2:
        q0, q1 = qubits

        alphas = (
            [alpha**2]
            + 2 * [alpha * gamma]
            + [alpha * beta]
            + 2 * [alpha * gamma]
            + [alpha * beta]
            + 2 * [gamma**2]
            + [beta * gamma]
            + 2 * [gamma**2]
            + 3 * [beta * gamma]
            + [beta**2]
        )
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
