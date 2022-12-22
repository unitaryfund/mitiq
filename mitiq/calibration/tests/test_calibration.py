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
from functools import partial

import cirq
import numpy as np
from schema import Schema, Or

from mitiq import Executor, MeasurementResult
from mitiq.benchmarks import generate_rb_circuits
from mitiq.calibration import Calibrator, Settings, ZNESettings
from mitiq.calibration.calibration import (
    bitstrings_to_distribution,
    convert_to_expval_executor,
)
from mitiq.calibration.settings import Strategy
from mitiq.zne.inference import LinearFactory, RichardsonFactory
from mitiq.zne.scaling import (
    fold_gates_at_random,
    fold_global,
)


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


settings = Settings(
    techniques=["zne"],
    circuit_types=["ghz", "rb"],
    num_qubits=2,
    circuit_depth=10,
    technique_params={
        "scale_factors": [[1.0, 1.2, 1.4], [1.0, 1.5, 2.0], [1.0, 2.0, 3.0]],
        "scale_methods": [
            fold_global,
            fold_gates_at_random,
        ],
        "factories": [RichardsonFactory, LinearFactory],
    },
)


def test_ZNE_workflow():
    cal = Calibrator(execute, ZNESettings)
    assert cal.get_cost() == {"noisy_executions": 24, "ideal_executions": 0}

    cal.run()
    assert len(cal.results) == 3
    assert cal.results[0]["mitigated_values"]["zne"]["improvement_factor"] >= 0
    assert isinstance(cal.best_strategy(cal.results), Strategy)


def test_get_cost():
    cal = Calibrator(execute, settings)
    cost = cal.get_cost()
    expected_cost = 2 * 3 * 2 * 2  # circuits * scale * methods * factories
    assert cost["noisy_executions"] == expected_cost
    assert cost["ideal_executions"] == 0


def test_validate_run_circuits_schema():
    cal = Calibrator(execute, settings)
    results = cal.run_circuits()
    results_schema = Schema(
        [
            {
                "circuit_info": {
                    "circuit_depth": int,
                    "ideal_distribution": {str: float},
                    "num_qubits": int,
                    "two_qubit_gate_count": int,
                    "type": str,
                },
                "ideal_value": float,
                "noisy_value": float,
                "mitigated_values": {
                    str: {
                        "improvement_factor": Or(float, None),
                        "results": [
                            {
                                "factory": str,
                                "scale_factors": [float],
                                "scale_method": str,
                                "mitigated_value": float,
                                "improvement_factor": float,
                                "strategy": Strategy,
                            }
                        ],
                    }
                },
            }
        ]
    )
    assert results_schema.validate(results)


def test_compute_improvements_modifies_IF():
    cal = Calibrator(execute, settings)
    cal.run_circuits()
    IFs = [
        res["mitigated_values"]["zne"]["improvement_factor"]
        for res in cal.results
    ]
    assert all(IF is None for IF in IFs)
    cal.compute_improvements(cal.results)
    IFs = [
        res["mitigated_values"]["zne"]["improvement_factor"]
        for res in cal.results
    ]
    assert all(IF is not None for IF in IFs)


def test_bitstrings_to_distribution():
    bitstrings = [[1, 1], [1, 1], [1, 1], [1, 0]]
    distribution = bitstrings_to_distribution(bitstrings)
    assert distribution == {"11": 0.75, "10": 0.25}

    bitstrings = [[0, 0, 0], [0, 0, 1], [0, 0, 1], [1, 0, 0], [0, 0, 1]]
    distribution = bitstrings_to_distribution(bitstrings)
    assert distribution == {"000": 0.2, "001": 0.6, "100": 0.2}


def test_bitstrings_to_distribution_normalized():
    bitstrings = np.random.randint(2, size=(50, 3))
    distribution = bitstrings_to_distribution(bitstrings)
    assert np.isclose(sum(distribution.values()), 1)


def test_convert_to_expval_executor():
    noiseless_bitstring_executor = Executor(partial(execute, noise_level=0))
    noiseless_expval_executor, _ = convert_to_expval_executor(
        noiseless_bitstring_executor, {"00": 1.0}
    )
    rb_circuit = generate_rb_circuits(2, 10)[0]
    rb_circuit.append(cirq.measure(rb_circuit.all_qubits()))

    rb_circuit_expval = noiseless_expval_executor.evaluate(rb_circuit)
    assert np.isclose(rb_circuit_expval, 1.0)


# def test_execute_with_mitigation():
