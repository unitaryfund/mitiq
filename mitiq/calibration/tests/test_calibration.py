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
import cirq
import numpy as np

from mitiq import MeasurementResult
from mitiq.calibration import Calibrator, ZNESettings


def execute(circuit, noise_level=0.001):
    circuit = circuit.with_noise(cirq.amplitude_damp(noise_level))

    result = cirq.DensityMatrixSimulator().run(circuit, repetitions=100)
    return MeasurementResult(
        result=np.column_stack(list(result.measurements.values())).tolist(),
        qubit_indices=tuple(
            # q[2:-1] is necessary to convert "q(number)" into "number"
            int(q[2:-1])
            for k in result.measurements.keys()
            for q in k.split(",")
        ),
    )


def test_workflow():
    cal = Calibrator(execute, ZNESettings)
    assert cal.get_cost() == {"noisy_executions": 24, "ideal_executions": 0}

    results = cal.run_circuits()
    cal.compute_improvements(results)
    assert cal.get_optimal_strategy(results) == "zne"
