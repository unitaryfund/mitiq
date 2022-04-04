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

from cirq import Circuit, X, Y, I, LineQubit  # , Z


def xx(
    slack_length: int, num_repetitions: int = 1, spacing: int = -1
) -> Circuit:
    """Returns an XX digital dynamical decoupling sequence, based on inputs.

    Args:
        slack_length: Length of idle window to fill.
        num_repetitions: How many repetitions of the dd rule to apply. Default
            to 1.
        spacing: How many identity spacing gates to apply between dynamical
            decoupling gates, as a positive-value int. Defaults to evenly
            spaced.
    Returns:
        An XX digital dynamical decoupling sequence, as a cirq circuit
    """
    default_length = 2
    num_decoupling_gates = default_length * num_repetitions
    slack_difference = slack_length - num_decoupling_gates
    if spacing < 0:
        spacing = slack_difference // (num_decoupling_gates + 1)
    q = LineQubit(0)
    slack_gates = [I(q) for _ in range(spacing)]
    ddd_circuit = Circuit(
        slack_gates,
        [
            (
                X(q),
                slack_gates,
            )
            for _ in range(num_decoupling_gates)
        ],
    )
    return ddd_circuit


def xyxy(
    slack_length: int, num_repetitions: int = 1, spacing: int = -1
) -> Circuit:
    """Returns an XYXY digital dynamical decoupling sequence, based on inputs.

    Args:
        slack_length: Length of idle window to fill.
        num_repetitions: How many repetitions of the dd rule to apply. Default
            to 1.
        spacing: How many identity spacing gates to apply between dynamical
            decoupling gates, as a positive-value int. Defaults to evenly
            spaced.
    Returns:
        An XYXY digital dynamical decoupling sequence, as a cirq circuit
    """
    default_length = 4
    num_decoupling_gates = default_length * num_repetitions
    slack_difference = slack_length - num_decoupling_gates
    if spacing < 0:
        spacing = slack_difference // (num_decoupling_gates + 1)
    q = LineQubit(0)
    slack_gates = [I(q) for _ in range(spacing)]
    ddd_circuit = Circuit(
        slack_gates,
        [
            (
                Y(q) if i % 2 else X(q),
                slack_gates,
            )
            for i in range(num_decoupling_gates)
        ],
    )
    return ddd_circuit


def yy(
    slack_length: int, num_repetitions: int = 1, spacing: int = -1
) -> Circuit:
    """Returns a YY digital dynamical decoupling sequence, based on inputs.

    Args:
        slack_length: Length of idle window to fill.
        num_repetitions: How many repetitions of the dd rule to apply. Default
            to 1.
        spacing: How many identity spacing gates to apply between dynamical
            decoupling gates, as a positive-value int. Defaults to evenly
            spaced.
    Returns:
        An YY digital dynamical decoupling sequence, as a cirq circuit
    """
    default_length = 2
    num_decoupling_gates = default_length * num_repetitions
    slack_difference = slack_length - num_decoupling_gates
    if spacing < 0:
        spacing = slack_difference // (num_decoupling_gates + 1)
    q = LineQubit(0)
    slack_gates = [I(q) for _ in range(spacing)]
    ddd_circuit = Circuit(
        slack_gates,
        [
            (
                Y(q),
                slack_gates,
            )
            for _ in range(num_decoupling_gates)
        ],
    )
    return ddd_circuit
