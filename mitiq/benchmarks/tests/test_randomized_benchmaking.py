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

"""Tests for randomized benchmarking with zero-noise extrapolation."""
import pytest
from itertools import product
import numpy as np

from mitiq.benchmarks.randomized_benchmarking import generate_rb_circuits
from mitiq.zne.inference import (
    LinearFactory,
    RichardsonFactory,
    PolyFactory,
    ExpFactory,
    AdaExpFactory,
)
from mitiq.zne.scaling import (
    fold_gates_at_random,
    fold_gates_from_left,
    fold_gates_from_right,
    fold_global,
)
from mitiq.benchmarks.utils import noisy_simulation
from mitiq.zne import mitigate_executor
from mitiq._typing import SUPPORTED_PROGRAM_TYPES

SCALE_FUNCTIONS = [
    fold_gates_at_random,
    fold_gates_from_left,
    fold_gates_from_right,
    fold_global,
]

FACTORIES = [
    AdaExpFactory(steps=3, scale_factor=1.5, asymptote=0.25),
    ExpFactory([1.0, 1.4, 2.1], asymptote=0.25),
    RichardsonFactory([1.0, 1.4, 2.1]),
    LinearFactory([1.0, 1.6]),
    PolyFactory([1.0, 1.4, 2.1], order=2),
]


@pytest.mark.parametrize("n_qubits", (1, 2))
def test_rb_circuits(n_qubits):
    depth = 10

    # test single qubit RB
    for trials in [2, 3]:
        circuits = generate_rb_circuits(
            n_qubits=n_qubits, num_cliffords=depth, trials=trials
        )
        for qc in circuits:
            # we check the ground state population to ignore any global phase
            wvf = qc.final_state_vector()
            zero_prob = abs(wvf[0] ** 2)
            assert np.isclose(zero_prob, 1)


@pytest.mark.parametrize(
    ["scale_noise", "fac"], product(SCALE_FUNCTIONS, FACTORIES)
)
def test_random_benchmarks(scale_noise, fac):
    trials = 3
    circuits = generate_rb_circuits(n_qubits=2, num_cliffords=4, trials=trials)
    noise = 0.01
    obs = np.diag([1, 0, 0, 0])

    def executor(qc):
        return noisy_simulation(qc, noise=noise, obs=obs)

    mit_executor = mitigate_executor(executor, fac, scale_noise)

    unmitigated = []
    mitigated = []
    for qc in circuits:
        unmitigated.append(executor(qc))
        mitigated.append(mit_executor(qc))

    unmit_err = np.abs(1.0 - np.asarray(unmitigated))
    mit_err = np.abs(1.0 - np.asarray(mitigated))

    assert np.average(unmit_err) >= np.average(mit_err)


@pytest.mark.parametrize("n_qubits", (1, 2))
@pytest.mark.parametrize("return_type", SUPPORTED_PROGRAM_TYPES.keys())
def test_rb_conversion(n_qubits, return_type):
    depth = 10
    for trials in [2, 3]:
        circuits = generate_rb_circuits(
            n_qubits=n_qubits,
            num_cliffords=depth,
            trials=trials,
            return_type=return_type,
        )
        for qc in circuits:
            assert return_type in qc.__module__
