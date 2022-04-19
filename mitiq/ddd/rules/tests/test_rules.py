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

"""Unit tests for DDD rules."""
from mitiq.ddd.rules.rules import general_rule, xx, xyxy, yy, repeated_rule
import pytest
from cirq import X, Y, Z, I, Circuit, LineQubit
from mitiq.utils import _equal


@pytest.mark.parametrize(
    "slack_length",
    [4, 5, 8, 13, 21, 34],
)
@pytest.mark.parametrize(
    "gates",
    [
        [X, X],
        [X, Y, X, Y],
        [Y, Y],
        [X, Y, Z],
    ],
)
def test_general_sequences(slack_length, gates):
    sequence = general_rule(
        slack_length=slack_length,
        gates=gates,
    )
    gate_set = {X, Y, Z}
    seq_gates = [op.gate for op in sequence.all_operations()]
    assert len(sequence) == slack_length
    assert gates == [gate for gate in seq_gates if gate in gate_set]


@pytest.mark.parametrize(
    "slack_length",
    [5, 8, 13, 21, 34],
)
@pytest.mark.parametrize(
    "rule",
    [
        xx,
        xyxy,
        yy,
    ],
)
def test_built_in_sequences(rule, slack_length):
    name = rule.__name__
    sequence = rule(slack_length)
    gates = [X if i == "x" else Y for i in name]
    gate_set = {X, Y}
    seq_gates = [op.gate for op in sequence.all_operations()]
    assert len(sequence) == slack_length
    assert gates == [gate for gate in seq_gates if gate in gate_set]


@pytest.mark.parametrize(
    ("slack_length", "rule", "sequence"),
    [
        (
            5,
            xx,
            Circuit(
                [
                    X(LineQubit(0)) if i % 2 else I(LineQubit(0))
                    for i in range(5)
                ]
            ),
        ),
        (
            5,
            yy,
            Circuit(
                [
                    Y(LineQubit(0)) if i % 2 else I(LineQubit(0))
                    for i in range(5)
                ]
            ),
        ),
        (
            4,
            xyxy,
            Circuit(
                [
                    Y(LineQubit(0)) if i % 2 else X(LineQubit(0))
                    for i in range(4)
                ]
            ),
        ),
    ],
)
def test_exact_sequences(slack_length, rule, sequence):
    sequence_to_test = rule(slack_length)
    assert _equal(sequence_to_test, sequence)


@pytest.mark.parametrize(
    "slack_length",
    [1, 2, 3, 5, 8, 13, 21, 34],
)
@pytest.mark.parametrize("spacing", [i for i in range(5, 7)])
def test_rule_failures(slack_length, spacing):
    num_decoupling_gates = 3
    if slack_length < num_decoupling_gates:
        sequence = general_rule(
            slack_length=slack_length,
            spacing=spacing,
            gates=[X, Y, Z],
        )
        assert len(sequence) == 0
    elif slack_length < (
        (num_decoupling_gates + 1) * spacing + num_decoupling_gates
    ):
        with pytest.raises(
            ValueError, match="too long for given slack window"
        ):
            general_rule(
                slack_length=slack_length,
                spacing=spacing,
                gates=[X, Y, Z],
            )
    else:
        sequence = general_rule(
            slack_length=slack_length,
            spacing=spacing,
            gates=[X, Y, Z],
        )
        assert len(sequence) == slack_length


@pytest.mark.parametrize(
    "slack_length",
    [1, 2, 3, 5],
)
@pytest.mark.parametrize(
    "gates",
    [
        [X],
        [Y],
        [Z],
    ],
)
def test_general_for_incomplete_rule(slack_length, gates):
    with pytest.raises(ValueError, match="too short to make a ddd sequence"):
        general_rule(
            slack_length=slack_length,
            gates=gates,
        )


@pytest.mark.parametrize(
    "slack_length",
    [4, 5, 8, 13, 21, 34],
)
@pytest.mark.parametrize(
    "gates",
    [
        [X, X],
        [X, Y, X, Y],
        [Y, Y],
        [X, Y, Z],
    ],
)
def test_repeated_sequences(slack_length, gates):
    sequence = repeated_rule(
        slack_length=slack_length,
        gates=gates,
    )
    num_reps = slack_length // len(gates)
    gate_set = {X, Y, Z}
    seq_gates = [op.gate for op in sequence.all_operations()]
    assert len(sequence) == slack_length
    assert gates * num_reps == [gate for gate in seq_gates if gate in gate_set]
