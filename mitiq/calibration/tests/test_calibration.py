# Copyright (C) Unitary Fund
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the Clifford data regression top-level API."""
from functools import partial
import pytest

import cirq
import numpy as np

from mitiq import Executor, MeasurementResult, SUPPORTED_PROGRAM_TYPES
from mitiq.benchmarks import generate_rb_circuits
from mitiq.calibration import (
    Calibrator,
    Settings,
    execute_with_mitigation,
)
from mitiq.calibration.calibrator import (
    convert_to_expval_executor,
    ExperimentResults,
    MissingResultsError,
)
from mitiq.calibration.settings import (
    PECSettings,
    Strategy,
    MitigationTechnique,
    BenchmarkProblem,
    ZNESettings,
)
from mitiq.zne.inference import LinearFactory, RichardsonFactory
from mitiq.zne.scaling import fold_global
from mitiq.interface import convert_to_mitiq
from mitiq.pec.representations import (
    represent_operation_with_local_depolarizing_noise,
)

light_zne_settings = Settings(
    [
        {
            "circuit_type": "mirror",
            "num_qubits": 1,
            "circuit_depth": 1,
        },
        {
            "circuit_type": "mirror",
            "num_qubits": 2,
            "circuit_depth": 1,
        },
    ],
    strategies=[
        {
            "technique": "zne",
            "scale_noise": fold_global,
            "factory": LinearFactory([1.0, 2.0]),
        },
    ],
)

light_pec_settings = Settings(
    [
        {
            "circuit_type": "mirror",
            "num_qubits": 1,
            "circuit_depth": 1,
        },
        {
            "circuit_type": "mirror",
            "num_qubits": 2,
            "circuit_depth": 1,
        },
    ],
    strategies=[
        {
            "technique": "pec",
            "representation_function": (
                represent_operation_with_local_depolarizing_noise
            ),
            "is_qubit_dependent": False,
            "noise_level": 0.001,
            "num_samples": 200,
        },
    ],
)


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


def damping_execute(circuit, noise_level=0.001):
    circuit = circuit.with_noise(cirq.amplitude_damp(noise_level))

    result = cirq.DensityMatrixSimulator().run(circuit, repetitions=100)
    bitstrings = np.column_stack(list(result.measurements.values()))
    return MeasurementResult(bitstrings)


def depolarizing_execute(circuit, noise_level=0.01):
    circuit = circuit.with_noise(cirq.depolarize(noise_level))

    result = cirq.DensityMatrixSimulator().run(circuit, repetitions=100)
    bitstrings = np.column_stack(list(result.measurements.values()))
    return MeasurementResult(bitstrings)


def non_cirq_damping_execute(circuit):
    # Ensure test circuits are converted to user's frontend by the Calibrator
    assert not isinstance(circuit, cirq.Circuit)
    circuit, circuit_type = convert_to_mitiq(circuit)
    # Pennylane and Braket conversions discard measurements so we re-append
    if circuit_type in ["braket", "pennylane"]:
        circuit.append(cirq.measure(q) for q in circuit.all_qubits())
    return damping_execute(circuit)


def non_cirq_depolarizing_execute(circuit):
    # Ensure test circuits are converted to user's frontend by the Calibrator
    assert not isinstance(circuit, cirq.Circuit)
    circuit, circuit_type = convert_to_mitiq(circuit)
    # Pennylane and Braket conversions discard measurements so we re-append
    if circuit_type in ["braket", "pennylane"]:
        circuit.append(cirq.measure(q) for q in circuit.all_qubits())
    return depolarizing_execute(circuit)


def test_ZNE_workflow():
    cal = Calibrator(damping_execute, frontend="cirq")
    cost = cal.get_cost()
    assert cost == {"noisy_executions": 8 * 3 * 4, "ideal_executions": 0}

    cal.run()
    assert isinstance(cal.results, ExperimentResults)
    assert isinstance(cal.best_strategy(), Strategy)


def test_PEC_workflow():
    cal = Calibrator(
        depolarizing_execute, frontend="cirq", settings=PECSettings
    )
    cost = cal.get_cost()
    assert cost == {"noisy_executions": 4 * 2 * 200, "ideal_executions": 0}

    cal.run()
    assert isinstance(cal.results, ExperimentResults)
    assert isinstance(cal.best_strategy(), Strategy)


@pytest.mark.parametrize("circuit_type", SUPPORTED_PROGRAM_TYPES.keys())
def test_ZNE_workflow_multi_platform(circuit_type):
    """Test the ZNE workflow runs with all possible frontends."""
    # Only test frontends different from cirq
    if circuit_type == "cirq":
        return

    cal = Calibrator(
        non_cirq_damping_execute,
        frontend=circuit_type,
        settings=light_zne_settings,
    )
    cost = cal.get_cost()
    assert cost == {"noisy_executions": 2 * 2, "ideal_executions": 0}
    cal.run()
    assert isinstance(cal.results, ExperimentResults)
    assert isinstance(cal.best_strategy(), Strategy)


@pytest.mark.parametrize("circuit_type", SUPPORTED_PROGRAM_TYPES.keys())
def test_PEC_workflow_multi_platform(circuit_type):
    """Test the PEC workflow runs with all possible frontends."""
    # Only test frontends different from cirq
    if circuit_type == "cirq":
        return

    cal = Calibrator(
        non_cirq_damping_execute,
        frontend=circuit_type,
        settings=light_pec_settings,
    )
    cost = cal.get_cost()
    assert cost == {"noisy_executions": 2 * 200, "ideal_executions": 0}
    cal.run()
    assert isinstance(cal.results, ExperimentResults)
    assert isinstance(cal.best_strategy(), Strategy)


def test_get_cost():
    cal = Calibrator(damping_execute, frontend="cirq", settings=settings)
    cost = cal.get_cost()
    expected_cost = 2 * 12  # circuits * num_experiments
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

    cal = Calibrator(
        damping_execute, frontend="cirq", settings=test_strategy_settings
    )
    cal.run()
    assert not np.isnan(cal.results.mitigated).all()

    strategy = cal.best_strategy()
    assert strategy.technique.name == "ZNE"


def test_convert_to_expval_executor():
    noiseless_bitstring_executor = Executor(
        partial(damping_execute, noise_level=0)
    )
    noiseless_expval_executor = convert_to_expval_executor(
        noiseless_bitstring_executor, bitstring="00"
    )
    rb_circuit = generate_rb_circuits(2, 10)[0]
    rb_circuit.append(cirq.measure(rb_circuit.all_qubits()))

    rb_circuit_expval = noiseless_expval_executor.evaluate(rb_circuit)
    assert np.isclose(rb_circuit_expval, 1.0)


def test_execute_with_mitigation(monkeypatch):
    cal = Calibrator(damping_execute, frontend="cirq")

    expval_executor = convert_to_expval_executor(
        Executor(damping_execute), bitstring="00"
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


def test_cal_execute_w_mitigation():
    cal = Calibrator(damping_execute, frontend="cirq")
    cal.run()

    expval_executor = convert_to_expval_executor(
        Executor(damping_execute), bitstring="00"
    )
    rb_circuit = generate_rb_circuits(2, 10)[0]
    rb_circuit.append(cirq.measure(rb_circuit.all_qubits()))

    expval = cal.execute_with_mitigation(rb_circuit, expval_executor)
    assert isinstance(expval, float)
    assert 0 <= expval <= 1.5


def test_double_run():
    cal = Calibrator(damping_execute, frontend="cirq")
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


@pytest.mark.parametrize("settings", [ZNESettings, PECSettings])
def test_logging(capfd, settings):
    cal = Calibrator(damping_execute, frontend="cirq", settings=settings)
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
