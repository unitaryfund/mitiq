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

from cirq import LineQubit

from mitiq import PauliString, Observable, QPROGRAM
from mitiq._typing import SUPPORTED_PROGRAM_TYPES
from mitiq.cdr import (
    execute_with_cdr,
    linear_fit_function_no_intercept,
    linear_fit_function,
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
            "random_state": 1,
        },
    ],
)
def test_execute_with_cdr(circuit_type, fit_function, kwargs):
    circuit = random_x_z_cnot_circuit(
        LineQubit.range(2), n_moments=5, random_state=1
    )
    circuit = convert_from_mitiq(circuit, circuit_type)
    obs = Observable(PauliString("IZ"), PauliString("ZZ"))

    true_value = obs.expectation(circuit, simulate)
    noisy_value = obs.expectation(circuit, execute)
    cdr_value = execute_with_cdr(
        circuit,
        execute,
        obs,
        simulator=simulate,
        num_training_circuits=4,
        fraction_non_clifford=0.5,
        fit_function=fit_function,
        kwargs=kwargs,
    )
    assert abs(cdr_value - true_value) < abs(noisy_value - true_value)


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
        num_training_circuits=4,
        fraction_non_clifford=0.5,
        scale_factors=[3],
    )
    assert abs(vncdr_value - true_value) < abs(noisy_value - true_value)


def test_no_num_fit_parameters_with_custom_fit_raises_error():
    with pytest.raises(
        ValueError, match="Must provide arg `num_fit_parameters`"
    ):
        execute_with_cdr(
            random_x_z_cnot_circuit(
                LineQubit.range(2), n_moments=2, random_state=1
            ),
            execute,
            observables=Observable(PauliString()),
            simulator=simulate,
            fit_function=lambda _: 1,
        )
