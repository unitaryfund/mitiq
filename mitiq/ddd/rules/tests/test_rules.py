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
from mitiq.ddd.rules.rules import construct_rule, xx, xyxy, yy
import pytest
from cirq import X, Y, Z

phi = (1 + 5**0.5) / 2.0


@pytest.mark.parametrize(
    "slack_length",
    [int(round((phi**n - (1 - phi) ** n) / 5**0.5)) for n in range(2, 10)],
)
def test_rules(slack_length):
    @pytest.mark.parametrize(
        "rule",
        [
            xx(slack_length),
            xyxy(slack_length),
            yy(slack_length),
        ],
    )
    def test_rule(rule):
        assert len(rule) == slack_length

    @pytest.mark.parametrize("spacing", [i for i in range(slack_length)])
    def user_spacing(spacing):
        @pytest.mark.parametrize(
            "gates",
            [
                [X, Y, X, Y],
                [Y, Y],
                [X, Y, Z],
            ],
        )
        def test_user_spaced_rule_construct(gates):
            rule = (
                construct_rule(
                    slack_length=slack_length,
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
                    spacing=spacing,
                ),
                xyxy(
                    slack_length=slack_length,
                    spacing=spacing,
                ),
                yy(
                    slack_length=slack_length,
                    spacing=spacing,
                ),
            ],
        )
        def test_user_spaced_rule(rule):
            assert len(rule) == slack_length


@pytest.mark.parametrize(
    "gates",
    [[X, Y, Z]],
)
@pytest.mark.parametrize(
    "slack_length",
    [int(round((phi**n - (1 - phi) ** n) / 5**0.5)) for n in range(2, 10)],
)
@pytest.mark.parametrize("spacing", [i for i in range(5, 7)])
def test_rule_failure(gates, slack_length, spacing):
    num_decoupling_gates = len(gates)
    if slack_length < (
        (num_decoupling_gates + 1) * spacing + num_decoupling_gates
    ):
        with pytest.raises(
            ValueError, match="too long for given slack window"
        ):
            construct_rule(
                slack_length=slack_length,
                spacing=spacing,
                gates=gates,
            )
