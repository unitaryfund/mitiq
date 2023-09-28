# Copyright (C) Unitary Fund
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

"""Tests for rotated randomized benchmarking circuits."""

import numpy as np
import pytest

from mitiq import SUPPORTED_PROGRAM_TYPES
from mitiq.benchmarks.rotated_randomized_benchmarking import (
    generate_rotated_rb_circuits,
)


@pytest.mark.parametrize("n_qubits", (1, 2))
@pytest.mark.parametrize("theta", np.pi * np.random.rand(3))
def test_rotated_rb_circuits(n_qubits, theta):
    depth = 10
    results = []
    for trials in [5, 10]:
        circuits = generate_rotated_rb_circuits(
            n_qubits=n_qubits, num_cliffords=depth, theta=theta, trials=trials
        )
        for qc in circuits:
            # we check the ground state population to ignore any global phase
            wvf = qc.final_state_vector()
            zero_prob = abs(wvf[0] ** 2)
            assert -1.0001 <= zero_prob <= 1.0001
            results.append(zero_prob)


@pytest.mark.parametrize("n_qubits", (1, 2))
@pytest.mark.parametrize("theta", np.pi * np.random.rand(3))
@pytest.mark.parametrize("return_type", SUPPORTED_PROGRAM_TYPES.keys())
def test_rotated_rb_conversion(n_qubits, theta, return_type):
    depth = 10
    for trials in [2, 3]:
        circuits = generate_rotated_rb_circuits(
            n_qubits=n_qubits,
            num_cliffords=depth,
            theta=theta,
            trials=trials,
            return_type=return_type,
        )
        for qc in circuits:
            assert return_type in qc.__module__
