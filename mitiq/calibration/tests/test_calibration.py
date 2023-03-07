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
import pytest

import cirq
import numpy as np

from mitiq import Executor, MeasurementResult
from mitiq.benchmarks import generate_rb_circuits
from mitiq.calibration import (
    Calibrator,
    Settings,
    ZNESettings,
    execute_with_mitigation,
)
from mitiq.calibration.calibrator import (
    convert_to_expval_executor,
    ExperimentResults,
    MissingResultsError,
)
from mitiq.calibration.settings import (
    Strategy,
    MitigationTechnique,
    BenchmarkProblem,
)
from mitiq.zne.inference import LinearFactory, RichardsonFactory
from mitiq.zne.scaling import fold_global


def execute(circuit, noise_level=0.001):
    circuit = circuit.with_noise(cirq.amplitude_damp(noise_level))

    result = cirq.DensityMatrixSimulator().run(circuit, repetitions=100)
    bitstrings = np.column_stack(list(result.measurements.values()))
    return MeasurementResult(bitstrings)


settings = Settings(
    [
        {
            "circuit_type": "ghz",
            "num_qubits": 2,
            "circuit_depth": 10,
        },
        {"circuit_type": "rb", "num_qubits": 2, "circuit_depth": 10},
    ],
    strategies=[
        {
            "technique": "zne",
            "scale_noise": fold_global,
            "factory": RichardsonFactory([1.0, 2.0, 3.0]),
        },
        {
            "technique": "zne",
            "scale_noise": fold_global,
            "factory": RichardsonFactory([1.0, 3.0, 5.0]),
        },
        {
            "technique": "zne",
            "scale_noise": fold_global,
            "factory": LinearFactory([1.0, 2.0, 3.0]),
        },
        {
            "technique": "zne",
            "scale_noise": fold_global,
            "factory": LinearFactory([1.0, 3.0, 5.0]),
        },
    ],
)


def test_ZNE_workflow():
    cal = Calibrator(execute, ZNESettings)
    cost = cal.get_cost()
    assert cost == {"noisy_executions": 24, "ideal_executions": 0}

    cal.run()
    num_strategies, num_problems = cal.results.mitigated.shape
    num_results = num_strategies * num_problems
    assert num_results == cost["noisy_executions"]
    assert isinstance(cal.results, ExperimentResults)
    assert isinstance(cal.best_strategy(), Strategy)


def test_get_cost():
    cal = Calibrator(execute, settings)
    cost = cal.get_cost()
    expected_cost = 2 * 4  # circuits * num_experiments
    assert cost["noisy_executions"] == expected_cost
    assert cost["ideal_executions"] == 0


def test_best_strategy():
    test_strategy_settings = Settings(
        benchmarks=[
            {"circuit_type": "ghz", "num_qubits": 2, "circuit_depth": 10},
            {
                "circuit_type": "mirror",
                "num_qubits": 2,
                "circuit_depth": 10,
                "circuit_seed": 1,
            },
        ],
        strategies=[
            {
                "technique": "zne",
                "scale_noise": fold_global,
                "factory": RichardsonFactory([1.0, 2.0, 3.0]),
            },
            {
                "technique": "zne",
                "scale_noise": fold_global,
                "factory": RichardsonFactory([1.0, 3.0, 5.0]),
            },
            {
                "technique": "zne",
                "scale_noise": fold_global,
                "factory": LinearFactory([1.0, 2.0, 3.0]),
            },
            {
                "technique": "zne",
                "scale_noise": fold_global,
                "factory": LinearFactory([1.0, 3.0, 5.0]),
            },
        ],
    )

    cal = Calibrator(execute, test_strategy_settings)
    cal.run()
    assert not np.isnan(cal.results.mitigated).all()

    strategy = cal.best_strategy()
    assert strategy.technique.name == "ZNE"


def test_convert_to_expval_executor():
    noiseless_bitstring_executor = Executor(partial(execute, noise_level=0))
    noiseless_expval_executor = convert_to_expval_executor(
        noiseless_bitstring_executor, bitstring="00"
    )
    rb_circuit = generate_rb_circuits(2, 10)[0]
    rb_circuit.append(cirq.measure(rb_circuit.all_qubits()))

    rb_circuit_expval = noiseless_expval_executor.evaluate(rb_circuit)
    assert np.isclose(rb_circuit_expval, 1.0)


def test_execute_with_mitigation(monkeypatch):
    cal = Calibrator(execute, ZNESettings)

    expval_executor = convert_to_expval_executor(
        Executor(execute), bitstring="00"
    )
    rb_circuit = generate_rb_circuits(2, 10)[0]
    rb_circuit.append(cirq.measure(rb_circuit.all_qubits()))

    # override the def of `input` so that it returns "yes"
    monkeypatch.setattr("builtins.input", lambda _: "yes")
    expval = execute_with_mitigation(
        rb_circuit, expval_executor, calibrator=cal
    )
    assert isinstance(expval, float)
    assert 0 <= expval <= 1.5


def test_double_run():
    cal = Calibrator(execute, ZNESettings)
    cal.run()
    cal.run()


def test_ExtrapolationResults_add_result():
    er = ExperimentResults(5, 3)
    assert er.is_missing_data()
    strat = Strategy(0, MitigationTechnique.ZNE, {})
    problem = BenchmarkProblem(0, "circuit", "ghz", {})
    er.add_result(
        strat, problem, ideal_val=1.0, noisy_val=0.8, mitigated_val=0.9
    )
    assert er.is_missing_data()
    er.mitigated = np.ones((5, 3))
    er.ideal = np.ones((5, 3))
    er.noisy = np.ones((5, 3))
    assert not er.is_missing_data()


def test_ExtrapolationResults_errors():
    num_strategies, num_problems = 5, 3
    er = ExperimentResults(num_strategies, num_problems)
    er.mitigated = np.random.random((num_strategies, num_problems))
    er.ideal = np.random.random((num_strategies, num_problems))

    assert (er.squared_errors() > 0).all()


def test_ExtrapolationResults_best_strategy():
    num_strategies, num_problems = 5, 3
    er = ExperimentResults(num_strategies, num_problems)
    er.mitigated = np.zeros((num_strategies, num_problems))
    er.mitigated[4, 2] = 0.8
    er.ideal = np.ones((num_strategies, num_problems))
    assert er.best_strategy_id() == 4


def test_logging(capfd):
    cal = Calibrator(execute, ZNESettings)
    cal.run(log=True)

    captured = capfd.readouterr()
    assert "circuit" in captured.out


def test_ExperimentResults_reset_data():
    num_strategies, num_problems = 5, 3
    er = ExperimentResults(num_strategies, num_problems)
    strat = Strategy(0, MitigationTechnique.ZNE, {})
    problem = BenchmarkProblem(0, "circuit", "ghz", {})
    er.add_result(
        strat, problem, ideal_val=1.0, noisy_val=0.8, mitigated_val=0.9
    )
    assert not np.isnan(er.mitigated).all()
    er.reset_data()
    assert np.isnan(er.mitigated).all()


def test_ExperimentResults_ensure_full():
    er = ExperimentResults(5, 3)
    with pytest.raises(MissingResultsError):
        er.ensure_full()
