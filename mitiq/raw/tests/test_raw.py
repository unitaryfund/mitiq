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

"""Unit tests for RAW (no error mitigation)."""

import functools

import numpy as np

import cirq

from mitiq import Executor, Observable, PauliString, raw
from mitiq.interface import mitiq_cirq


def test_raw():
    circuit = cirq.Circuit(cirq.H.on(cirq.LineQubit(0)))
    observable = Observable(PauliString("X"))
    executor = Executor(mitiq_cirq.compute_density_matrix)
    raw_value = raw.execute(circuit, executor, observable)

    assert isinstance(raw_value, complex)
    assert executor.executed_circuits == [circuit]

    compute_density_matrix_noiseless = functools.partial(
        mitiq_cirq.compute_density_matrix, noise_level=(0.0,)
    )
    executor_noiseless = Executor(compute_density_matrix_noiseless)
    true_value = raw.execute(circuit, executor_noiseless, observable)
    assert np.isclose(true_value, 1.0)
    assert executor_noiseless.executed_circuits == [circuit]
