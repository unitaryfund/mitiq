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

"""Test executor generator and example expectation functions in executor.py."""
from pyquil import get_qc, Program
from pyquil.gates import X

from mitiq.mitiq_pyquil.pyquil_utils import (
    generate_qcs_executor,
    ground_state_expectation,
)

QVM = get_qc("2q-qvm")


def test_ground_state_executor():
    executor = generate_qcs_executor(QVM, ground_state_expectation)
    program = Program(X(0))
    assert 0.0 == executor(program)

    # works for any number of qubits
    program = Program(X(0), X(1))
    assert 0.0 == executor(program)

    # when programs compose to the identity, we should get 1.0
    program = Program(X(0), X(0), X(1), X(1))
    assert 1.0 == executor(program)
