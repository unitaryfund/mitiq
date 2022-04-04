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
from mitiq.ddd.rules.rules import construct_rule, xx, xyxy, yy, random
import pytest
from cirq import X, Y, Z


@pytest.mark.parametrize("slack_length", [i for i in range(2, 25)])
@pytest.mark.parametrize("num_repetitions", [i for i in range(1, 5)])
def test_rules(slack_length, num_repetitions):
    @pytest.mark.parametrize(
        "rule",
        [
            xx(slack_length, num_repetitions),
            xyxy(slack_length, num_repetitions),
            yy(slack_length, num_repetitions),
        ],
    )
    def test_rule(rule):
        assert len(rule) == slack_length

    @pytest.mark.parametrize(
        "rule",
        [
            random(
                slack_length,
                sequence_length=[i for i in range(slack_length)][0],
                num_repetitions=num_repetitions,
            ),
        ],
    )
    def test_random_rule(rule):
        assert len(rule) == slack_length

    @pytest.mark.parametrize("spacing", [i for i in range(slack_length)])
    def user_spacing(spacing):
        @pytest.mark.parametrize(
            "gates",
            [
                [X, X],
                [X, Y, X, Y],
                [Y, Y],
                [Y, X, Y, X],
                [Z, X, Y, Z],
            ],
        )
        def test_user_spaced_rule_construct(gates):
            rule = (
                construct_rule(
                    slack_length=slack_length,
                    num_repetitions=num_repetitions,
                    spacing=spacing,
                    gates=gates,
                ),
            )
            assert len(rule) == slack_length

        @pytest.mark.parametrize(
            "rule",
            [
                xx(
                    slack_length=slack_length,
                    num_repetitions=num_repetitions,
                    spacing=spacing,
                ),
                xyxy(
                    slack_length=slack_length,
                    num_repetitions=num_repetitions,
                    spacing=spacing,
                ),
                yy(
                    slack_length=slack_length,
                    num_repetitions=num_repetitions,
                    spacing=spacing,
                ),
            ],
        )
        def test_user_spaced_rule(rule):
            assert len(rule) == slack_length

        @pytest.mark.parametrize(
            "rule",
            [
                random(
                    slack_length=slack_length,
                    sequence_length=[i for i in range(1, slack_length + 1)][0],
                    num_repetitions=num_repetitions,
                    spacing=spacing,
                ),
            ],
        )
        def test_user_spaced_random_rule(rule):
            assert len(rule) == slack_length


@pytest.mark.parametrize("num_repetitions", [i for i in range(-5, 0)])
@pytest.mark.parametrize(
    "gates",
    [
        [X, X],
        [X, Y, X, Y],
        [Y, Y],
        [Y, X, Y, X],
        [Z, X, Y, Z],
    ],
)
def test_negative_reps(num_repetitions, gates):
    with pytest.raises(ValueError, match="number of sequence repetitions"):
        construct_rule(
            slack_length=10,
            num_repetitions=num_repetitions,
            spacing=-1,
            gates=gates,
        )


@pytest.mark.parametrize(
    "gates",
    [
        [X, X],
        [X, Y, X, Y],
        [Y, Y],
        [Y, X, Y, X],
        [Z, X, Y, Z],
    ],
)
@pytest.mark.parametrize("slack_length", [i for i in range(8, 20)])
@pytest.mark.parametrize("num_repetitions", [i for i in range(2, 4)])
@pytest.mark.parametrize("spacing", [i for i in range(3, 7)])
def test_rule_failure(gates, slack_length, num_repetitions, spacing):
    rule_length = len(gates)
    num_decoupling_gates = rule_length * num_repetitions
    if slack_length < (
        (num_decoupling_gates + 1) * spacing + num_decoupling_gates
    ):
        with pytest.raises(
            ValueError, match="too long for given slack window"
        ):
            construct_rule(
                slack_length=slack_length,
                num_repetitions=num_repetitions,
                spacing=spacing,
                gates=gates,
            )
