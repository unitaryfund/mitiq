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
from cirq import Circuit, X, Y, I, LineQubit, Gate, unitary, is_unitary
from typing import List
from itertools import cycle


def general_rule(
    slack_length: int, gates: List[Gate], spacing: int = -1
) -> Circuit:
    """Returns a digital dynamical decoupling sequence, based on inputs.

    Args:
        slack_length: Length of idle window to fill.
        spacing: How many identity spacing gates to apply between dynamical
            decoupling gates, as a non-negative int. Negative int corresponds
            to default. Defaults to maximal spacing that fits a single sequence
            in the given slack window.
            E.g. given slack_length = 20, gates = [X, X] the spacing defaults
            to 6 and the rule returns the sequence:
            ──I──I──I──I──I──I──X──I──I──I──I──I──I──X──I──I──I──I──I──I──
            given slack_length = 20, gates [X, Y, X, Y] the spacing defaults
            to 3 and the rule returns the sequence:
            ──I──I──I──I──X──I──I──I──Y──I──I──I──X──I──I──I──Y──I──I──I──
        gates: A list of Cirq gates to build the rule. E.g. [X, X] is the xx
            sequence, [X, Y, X, Y] is the xyxy sequence.
            - Note: To repeat the sequence, specify a repeated gateset.
    Returns:
        A digital dynamical decoupling sequence, as a Cirq circuit.
    """
    if len(gates) < 2:
        raise ValueError("Gateset too short to make a ddd sequence.")
    if slack_length < 2 or slack_length < len(gates):
        return Circuit()
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
    sequence = Circuit(
        slack_gates,
        [
            (
                gate.on(q),
                slack_gates,
            )
            for (_, gate) in zip(range(num_decoupling_gates), cycle(gates))
        ],
    )
    if not is_unitary(unitary(sequence)):
        raise ValueError("Provided sequence is not unitary!")
    for i in range(slack_remainder):
        sequence.append(I(q)) if i % 2 else sequence.insert(0, I(q))
    return sequence


def xx(slack_length: int, spacing: int = -1) -> Circuit:
    """Returns an XX digital dynamical decoupling sequence, based on inputs.

    Args:
        slack_length: Length of idle window to fill.
        spacing: How many identity spacing gates to apply between dynamical
            decoupling gates, as a non-negative int. Negative int corresponds
            to default. Defaults to maximal spacing that fits a single sequence
            in the given slack window.
            E.g. given slack_length = 20 the spacing defaults to 6 and this
            rule returns the sequence:
            ──I──I──I──I──I──I──X──I──I──I──I──I──I──X──I──I──I──I──I──I──
    Returns:
        An XX digital dynamical decoupling sequence, as a Cirq circuit.
    """
    xx_sequence = general_rule(
        slack_length=slack_length,
        spacing=spacing,
        gates=[X, X],
    )
    return xx_sequence


def xyxy(slack_length: int, spacing: int = -1) -> Circuit:
    """Returns an XYXY digital dynamical decoupling sequence, based on inputs.

    Args:
        slack_length: Length of idle window to fill.
        spacing: How many identity spacing gates to apply between dynamical
            decoupling gates, as a non-negative int. Negative int corresponds
            to default. Defaults to maximal spacing that fits a single sequence
            in the given slack window.
            E.g. given slack_length = 20 the spacing defaults to 3 and this
            rule returns the sequence:
            ──I──I──I──I──X──I──I──I──Y──I──I──I──X──I──I──I──Y──I──I──I──
    Returns:
        An XYXY digital dynamical decoupling sequence, as a Cirq circuit.
    """
    xyxy_sequence = general_rule(
        slack_length=slack_length,
        spacing=spacing,
        gates=[X, Y, X, Y],
    )
    return xyxy_sequence


def yy(slack_length: int, spacing: int = -1) -> Circuit:
    """Returns a YY digital dynamical decoupling sequence, based on inputs.

    Args:
        slack_length: Length of idle window to fill.
        spacing: How many identity spacing gates to apply between dynamical
            decoupling gates, as a non-negative int. Negative int corresponds
            to default. Defaults to maximal spacing that fits a single sequence
            in the given slack window.
            E.g. given slack_length = 20 the spacing defaults to 6 and
            this rule returns the sequence:
            ──I──I──I──I──I──I──Y──I──I──I──I──I──I──Y──I──I──I──I──I──I──
    Returns:
        An YY digital dynamical decoupling sequence, as a Cirq circuit.
    """
    yy_sequence = general_rule(
        slack_length=slack_length,
        spacing=spacing,
        gates=[Y, Y],
    )
    return yy_sequence


def repeated_rule(slack_length: int, gates: List[Gate]) -> Circuit:
    """Returns a general digital dynamical decoupling sequence that repeats
    until the slack is filled without spacing, up to a complete repetition.

    Note:
    Args:
        slack_length: Length of idle window to fill.
        gates: A list of Cirq gates to build the rule. E.g. [X, X] is the xx
            sequence, [X, Y, X, Y] is the xyxy sequence.
    Returns:
        A repeated digital dynamical decoupling sequence, as a Cirq circuit.
    Note:
        Where the general rule requires pre-knowledge of the amount of slack,
        this rule attempts to fill every moment with sequence repetitions
        (up to a complete repetition of the gate set).
        E.g. given slack_length = 17 and gates = [X, Y]
            this rule returns the sequence:
            ──I──X──Y──X──Y──X──Y──X──Y──X──Y──X──Y──X──Y──X──Y──
    """
    num_decoupling_gates = len(gates)
    sequence = general_rule(
        slack_length=slack_length,
        spacing=0,
        gates=[
            gate
            for (_, gate) in zip(
                range(slack_length - (slack_length % num_decoupling_gates)),
                cycle(gates),
            )
        ],
    )
    return sequence
