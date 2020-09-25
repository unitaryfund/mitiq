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

"""Tests related to mitiq.pec.utils functions."""

from pytest import mark
import numpy as np
from cirq import Gate, LineQubit, X, Y, Z, I
from mitiq.pec.utils import _simple_pauli_deco_dict

BASE_NOISE = 0.01
DECO_DICT = _simple_pauli_deco_dict(base_noise=BASE_NOISE)


@mark.parametrize("gate", [X, Y, Z])
def test_simple_pauli_deco_dict(gate: Gate):
    epsilon = BASE_NOISE * 4.0 / 3.0
    c_neg = - (1 / 4) * epsilon / (1 - epsilon)
    c_pos = 1 - 3 * c_neg
    qreg = LineQubit.range(2)
    for q in qreg:

        x_deco = DECO_DICT[gate.on(q)]
        first_coefficient, first_imp_seq = x_deco[0]
        assert np.isclose(c_pos, first_coefficient)
        assert first_imp_seq == [gate.on(q)]
        second_coefficient, second_imp_seq = x_deco[1]
        assert np.isclose(c_neg, second_coefficient)
        assert second_imp_seq == [gate.on(q), X.on(q)]


@mark.parametrize("gate", [X, Y, Z])
def test_simplify_paulist_in_simple_pauli_deco_dict(gate: Gate):
    qreg = LineQubit.range(2)
    deco_dict = _simple_pauli_deco_dict(BASE_NOISE, simplify_paulis=True)
    for q in qreg:
        x_deco = deco_dict[gate.on(q)]
        _, first_imp_seq = x_deco[0]
        assert first_imp_seq == [gate.on(q)]
        _, second_imp_seq = x_deco[1]
        input_times_x = {X: I, Y: Z, Z: Y}
        assert second_imp_seq == [input_times_x[gate].on(q)]
