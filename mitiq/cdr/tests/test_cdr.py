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

import numpy as np

import cirq
from cirq import LineQubit

from mitiq import PauliString, Observable, QPROGRAM
from mitiq._typing import SUPPORTED_PROGRAM_TYPES
from mitiq.cdr import (
    execute_with_cdr,
    linear_fit_function_no_intercept,
    linear_fit_function,
    mitigate_executor,
    cdr_decorator,
)

from mitiq.interface import convert_from_mitiq, convert_to_mitiq

from mitiq.cdr._testing import random_x_z_cnot_circuit
from mitiq.interface.mitiq_cirq import compute_density_matrix


# Allow execution with any QPROGRAM for testing.
def execute(circuit: QPROGRAM) -> np.ndarray:
    return compute_density_matrix(convert_to_mitiq(circuit)[0])


def simulate(circuit: QPROGRAM) -> np.ndarray:
    return compute_density_matrix(
        convert_to_mitiq(circuit)[0], noise_level=(0,)
    )


@pytest.mark.parametrize("circuit_type", SUPPORTED_PROGRAM_TYPES.keys())
@pytest.mark.parametrize(
    "fit_function", [linear_fit_function, linear_fit_function_no_intercept]
)
@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {
            "method_select": "gaussian",
            "method_replace": "gaussian",
            "sigma_select": 0.5,
            "sigma_replace": 0.5,
        },
    ],
)
@pytest.mark.parametrize("random_state", [1, 2, 3, 4, 5])
def test_execute_with_cdr(circuit_type, fit_function, kwargs, random_state):
    circuit = random_x_z_cnot_circuit(
        LineQubit.range(2),
        n_moments=5,
        random_state=random_state,
    )
    circuit = convert_from_mitiq(circuit, circuit_type)
    obs = Observable(PauliString("XZ"), PauliString("YY"))

    true_value = obs.expectation(circuit, simulate)
    noisy_value = obs.expectation(circuit, execute)

    cdr_value = execute_with_cdr(
        circuit,
        execute,
        obs,
        simulator=simulate,
        num_training_circuits=20,
        fraction_non_clifford=0.5,
        fit_function=fit_function,
        random_state=random_state,
        **kwargs,
    )
    assert abs(cdr_value - true_value) <= abs(noisy_value - true_value)


@pytest.mark.parametrize("circuit_type", SUPPORTED_PROGRAM_TYPES.keys())
@pytest.mark.parametrize(
    "fit_function", [linear_fit_function, linear_fit_function_no_intercept]
)
@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {
            "method_select": "gaussian",
            "method_replace": "gaussian",
            "sigma_select": 0.5,
            "sigma_replace": 0.5,
        },
    ],
)
@pytest.mark.parametrize("random_state", [1, 2, 3, 4, 5])
def test_decorator_execute_with_cdr(
    circuit_type, fit_function, kwargs, random_state
):
    obs = Observable(PauliString("XZ"), PauliString("YY"))

    @cdr_decorator(
        observable=obs,
        simulator=simulate,
        num_training_circuits=20,
        fraction_non_clifford=0.5,
        fit_function=fit_function,
        random_state=random_state,
        **kwargs,
    )
    def decorated_execute(circuit: QPROGRAM) -> np.ndarray:
        return execute(circuit)

    def decorated_execute_with_cdr(circuit_type):
        circuit = random_x_z_cnot_circuit(
            LineQubit.range(2),
            n_moments=5,
            random_state=random_state,
        )
        circuit = convert_from_mitiq(circuit, circuit_type)

        true_value = obs.expectation(circuit, simulate)
        noisy_value = obs.expectation(circuit, execute)

        cdr_value = decorated_execute(
            circuit,
        )
        assert abs(cdr_value - true_value) <= abs(noisy_value - true_value)

    def decorated_execute_using_clifford_circuit():
        a, b = cirq.LineQubit.range(2)
        clifCirc = cirq.Circuit(
            cirq.H.on(a),
            cirq.H.on(b),
        )
        cdr_mitigaged = decorated_execute(clifCirc)
        assert obs.expectation(clifCirc, simulate) == cdr_mitigaged

    decorated_execute_with_cdr(circuit_type)
    decorated_execute_using_clifford_circuit()


@pytest.mark.parametrize("circuit_type", SUPPORTED_PROGRAM_TYPES.keys())
@pytest.mark.parametrize(
    "fit_function", [linear_fit_function, linear_fit_function_no_intercept]
)
@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {
            "method_select": "gaussian",
            "method_replace": "gaussian",
            "sigma_select": 0.5,
            "sigma_replace": 0.5,
        },
    ],
)
@pytest.mark.parametrize("random_state", [1, 2, 3, 4, 5])
def test_mitigated_execute_with_cdr(
    circuit_type, fit_function, kwargs, random_state
):
    circuit = random_x_z_cnot_circuit(
        LineQubit.range(2),
        n_moments=5,
        random_state=random_state,
    )
    circuit = convert_from_mitiq(circuit, circuit_type)
    obs = Observable(PauliString("XZ"), PauliString("YY"))

    true_value = obs.expectation(circuit, simulate)
    noisy_value = obs.expectation(circuit, execute)

    cdr_executor = mitigate_executor(
        executor=execute,
        observable=obs,
        simulator=simulate,
        num_training_circuits=20,
        fraction_non_clifford=0.5,
        fit_function=fit_function,
        random_state=random_state,
        **kwargs,
    )
    cdr_mitigated = cdr_executor(circuit)
    assert abs(cdr_mitigated - true_value) <= abs(noisy_value - true_value)


@pytest.mark.parametrize("circuit_type", SUPPORTED_PROGRAM_TYPES.keys())
def test_execute_with_variable_noise_cdr(circuit_type):
    circuit = random_x_z_cnot_circuit(
        LineQubit.range(2), n_moments=5, random_state=1
    )
    circuit = convert_from_mitiq(circuit, circuit_type)
    obs = Observable(PauliString("IZ"), PauliString("ZZ"))

    true_value = obs.expectation(circuit, simulate)
    noisy_value = obs.expectation(circuit, execute)
    vncdr_value = execute_with_cdr(
        circuit,
        execute,
        obs,
        simulator=simulate,
        num_training_circuits=10,
        fraction_non_clifford=0.5,
        scale_factors=[1, 3],
        random_state=1,
    )
    assert abs(vncdr_value - true_value) <= abs(noisy_value - true_value)


@pytest.mark.parametrize("circuit_type", SUPPORTED_PROGRAM_TYPES.keys())
def test_mitigate_executor_with_variable_noise_cdr(circuit_type):
    circuit = random_x_z_cnot_circuit(
        LineQubit.range(2), n_moments=5, random_state=1
    )
    circuit = convert_from_mitiq(circuit, circuit_type)
    obs = Observable(PauliString("IZ"), PauliString("ZZ"))

    true_value = obs.expectation(circuit, simulate)
    noisy_value = obs.expectation(circuit, execute)
    vncdr_executor = mitigate_executor(
        executor=execute,
        observable=obs,
        simulator=simulate,
        num_training_circuits=10,
        fraction_non_clifford=0.5,
        scale_factors=[1, 3],
        random_state=1,
    )
    mitigated = vncdr_executor(circuit)
    assert abs(mitigated - true_value) <= abs(noisy_value - true_value)


def test_no_num_fit_parameters_with_custom_fit_raises_error():
    with pytest.raises(ValueError, match="Must provide `num_fit_parameters`"):
        execute_with_cdr(
            random_x_z_cnot_circuit(
                LineQubit.range(2), n_moments=2, random_state=1
            ),
            execute,
            observables=Observable(PauliString()),
            simulator=simulate,
            fit_function=lambda _: 1,
        )


def test_no_num_fit_parameters_mitigate_executor_raises_error():
    with pytest.raises(ValueError, match="Must provide `num_fit_parameters`"):
        mitigated_executor = mitigate_executor(
            executor=execute,
            observables=Observable(PauliString()),
            simulator=simulate,
            fit_function=lambda _: 1,
        )
        mitigated = (
            mitigated_executor(
                random_x_z_cnot_circuit(
                    LineQubit.range(2), n_moments=2, random_state=1
                )
            ),
        )
        mitigated


def test_execute_with_cdr_using_clifford_circuit():
    a, b = cirq.LineQubit.range(2)
    clifCirc = cirq.Circuit(
        cirq.H.on(a),
        cirq.H.on(b),
    )
    obs = Observable(PauliString("XZ"), PauliString("YY"))
    cdr_value = execute_with_cdr(
        clifCirc, observable=obs, executor=execute, simulator=simulate
    )
    assert obs.expectation(clifCirc, simulate) == cdr_value


def test_mitigate_executor_with_cdr_using_clifford_circuit():
    a, b = cirq.LineQubit.range(2)
    clifCirc = cirq.Circuit(
        cirq.H.on(a),
        cirq.H.on(b),
    )
    obs = Observable(PauliString("XZ"), PauliString("YY"))
    mitigated_executor = mitigate_executor(
        observable=obs, executor=execute, simulator=simulate
    )
    mitigated = mitigated_executor(clifCirc)
    assert obs.expectation(clifCirc, simulate) == mitigated
