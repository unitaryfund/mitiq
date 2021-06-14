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

import numpy as np

from cirq import LineQubit

from mitiq._typing import SUPPORTED_PROGRAM_TYPES
from mitiq.interface import convert_from_mitiq
from mitiq.cdr import execute_with_cdr, linear_fit_function_no_intercept
from mitiq.cdr.execute import calculate_observable
from mitiq.cdr._testing import (
    random_x_z_circuit,
    executor,
    simulator_statevector,
)
from mitiq.zne.scaling import fold_gates_from_left


@pytest.mark.parametrize("circuit_type", SUPPORTED_PROGRAM_TYPES.keys())
def test_execute_with_cdr(circuit_type):
    circuit = random_x_z_circuit(
        LineQubit.range(2), n_moments=2, random_state=1
    )
    circuit = convert_from_mitiq(circuit, circuit_type)

    # Define observables for testing.
    sigma_z = np.diag([1, -1])
    obs = np.kron(np.identity(2), sigma_z)
    obs2 = np.kron(sigma_z, sigma_z)
    obs_list = [np.diag(obs), np.diag(obs2)]

    exact_solution = [
        calculate_observable(simulator_statevector(circuit), observable=obs)
        for obs in obs_list
    ]

    kwargs = {
        "method_select": "gaussian",
        "method_replace": "gaussian",
        "sigma_select": 0.5,
        "sigma_replace": 0.5,
        "random_state": 1,
    }
    num_circuits = 4
    frac_non_cliff = 0.5
    noisy_executor = partial(executor, noise_level=0.5)
    results0 = execute_with_cdr(
        circuit,
        noisy_executor,
        simulator_statevector,
        obs_list,
        num_circuits,
        frac_non_cliff,
        full_output=True,
    )
    results1 = execute_with_cdr(
        circuit,
        noisy_executor,
        simulator_statevector,
        obs_list,
        num_circuits,
        frac_non_cliff,
        ansatz=linear_fit_function_no_intercept,
        num_fit_parameters=1,
        scale_noise=fold_gates_from_left,
        scale_factors=[3],
        full_output=True,
        **kwargs,
    )
    for results in [results0, results1]:
        for i in range(len(results[1])):
            assert abs(results[1][i][0] - exact_solution[i]) >= abs(
                results[0][i] - exact_solution[i]
            )
