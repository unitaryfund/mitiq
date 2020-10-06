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
from cirq import Gate, LineQubit, X, Y, Z, Operation

from mitiq.pec.utils import _simple_pauli_deco_dict
from mitiq.pec.sampling import sample_sequence

BASE_NOISE = 0.01
DECO_DICT = _simple_pauli_deco_dict(base_noise=BASE_NOISE)


@mark.parametrize("gate", [X, Y, Z])
def test_gsample_sequence(gate: Gate):
    q = LineQubit(0)
    for _ in range(1000):
        imp_seq, sign, norm = sample_sequence(gate.on(q), DECO_DICT)
        assert all([isinstance(op, Operation) for op in imp_seq])
        assert sign in {1.0, -1.0}
        assert norm > 1
