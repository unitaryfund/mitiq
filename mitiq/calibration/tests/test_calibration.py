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

"""Tests for the Clifford data regression top-level API."""
import pytest
import cirq
from mitiq.calibration import Calibration, ZNESettings


def execute(circuit, noise_level=0.001):
    noisy_circuit = circuit.with_noise(cirq.depolarize(p=noise_level))
    return (
        cirq.DensityMatrixSimulator()
        .simulate(noisy_circuit)
        .final_density_matrix[0, 0]
        .real
    )


def test_workflow():
    cal = Calibration(execute, ZNESettings)
    assert cal.get_cost() == {"noisy_executions": 24, "ideal_executions": 0}

    cal.run_circuits()
    cal.compute_improvements()
    assert cal.get_optimal_strategy() == "zne"
