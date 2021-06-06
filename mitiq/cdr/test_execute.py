# Copyright (C) 2021 Unitary Fund
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

"""Tests for the data regression portion of Clifford data regression."""
import pytest
import numpy as np

import cirq

from mitiq.cdr.execute import calculate_observable, normalize_measurements
from mitiq.cdr._testing import simulator_statevector, simulator

# Observables.
sigma_z = np.diag(np.diag([1, -1]))


@pytest.mark.parametrize("op_and_expectation_value", ((cirq.I, 1.0), (cirq.H, 0.0), (cirq.X, -1.0)))
def test_calculate_observable_sigmaz(op_and_expectation_value):
    """Tests <psi|Z|psi> is correct for |psi> \in {|0>, |+>, |1>}."""
    op, expected = op_and_expectation_value
    circuit = cirq.Circuit(op.on(cirq.LineQubit(0)))
    assert np.isclose(
        calculate_observable(simulator_statevector(circuit), sigma_z), expected,
        atol=1e-7
    )

    assert np.isclose(
        calculate_observable(simulator(circuit, shots=10_000), sigma_z), expected,
        atol=1e-2
    )


def test_dictionary_to_probabilities():
    counts = {bin(0): 2, bin(1): 3}
    normalized_counts = normalize_measurements(counts)
    assert normalized_counts == {bin(0): 2 / 5, bin(1): 3 / 5}
