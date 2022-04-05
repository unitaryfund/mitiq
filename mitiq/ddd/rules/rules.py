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

"""Built-in rules determining what DDD sequence should be applied in a given
slack window.
"""
from cirq import Circuit, X, Y, I, Z, LineQubit, Gate
from typing import List, Optional

from itertools import cycle
import numpy as np


def construct_rule(
    slack_length: int, spacing: int, gates: List[Gate]
) -> Circuit:
    """Returns a digital dynamical decoupling sequence, based on inputs.

    Args:
        slack_length: Length of idle window to fill.
        spacing: How many identity spacing gates to apply between dynamical
            decoupling gates, as a non-negative int. Defaults to evenly
            spaced.
        gates: A list of Cirq gates to build the rule. E.g. [X, X] is the xx
            sequence, [X, Y, X, Y] is the xyxy sequence
            - Note: To repeat the sequence, specify a repeated gateset
    Returns:
        A digital dynamical decoupling sequence, as a cirq circuit
    """
    num_decoupling_gates = len(gates)
    slack_difference = slack_length - num_decoupling_gates
    if spacing < 0:
        spacing = slack_difference // (num_decoupling_gates + 1)
    slack_remainder = slack_length - (
        spacing * (num_decoupling_gates + 1) + num_decoupling_gates
    )
    if slack_remainder < 0:
        raise ValueError(
            "Rule too long for given slack window by {} moments.".format(
                slack_remainder * -1
            )
        )
    q = LineQubit(0)
    slack_gates = [I(q) for _ in range(spacing)]
    ddd_circuit = Circuit(
        slack_gates,
        [
            (
                gate.on(q),
                slack_gates,
            )
            for (_, gate) in zip(range(num_decoupling_gates), cycle(gates))
        ],
    )
    for i in range(slack_remainder):
        ddd_circuit.append(I(q)) if i % 2 else ddd_circuit.insert(0, I(q))
    return ddd_circuit


def xx(slack_length: int, spacing: int = -1) -> Circuit:
    """Returns an XX digital dynamical decoupling sequence, based on inputs.

    Args:
        slack_length: Length of idle window to fill.
        spacing: How many identity spacing gates to apply between dynamical
            decoupling gates, as a non-negative int. Defaults to evenly
            spaced.
    Returns:
        An XX digital dynamical decoupling sequence, as a cirq circuit
    """
    xx_rule = construct_rule(
        slack_length=slack_length,
        spacing=spacing,
        gates=[X, X],
    )
    return xx_rule


def xyxy(slack_length: int, spacing: int = -1) -> Circuit:
    """Returns an XYXY digital dynamical decoupling sequence, based on inputs.

    Args:
        slack_length: Length of idle window to fill.
        spacing: How many identity spacing gates to apply between dynamical
            decoupling gates, as a non-negative int. Defaults to evenly
            spaced.
    Returns:
        An XYXY digital dynamical decoupling sequence, as a cirq circuit
    """
    xyxy_rule = construct_rule(
        slack_length=slack_length,
        spacing=spacing,
        gates=[X, Y, X, Y],
    )
    return xyxy_rule


def yy(slack_length: int, spacing: int = -1) -> Circuit:
    """Returns a YY digital dynamical decoupling sequence, based on inputs.

    Args:
        slack_length: Length of idle window to fill.
        spacing: How many identity spacing gates to apply between dynamical
            decoupling gates, as a non-negative int. Defaults to evenly
            spaced.
    Returns:
        An YY digital dynamical decoupling sequence, as a cirq circuit
    """
    yy_rule = construct_rule(
        slack_length=slack_length,
        spacing=spacing,
        gates=[Y, Y],
    )
    return yy_rule

