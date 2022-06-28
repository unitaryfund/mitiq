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
""" Tests for Cirq executors defined in cirq_utils.py"""

import numpy as np
import cirq

from mitiq.interface.mitiq_cirq import (
    sample_bitstrings,
    compute_density_matrix,
    execute_with_depolarizing_noise,
)
from mitiq.rem.measurement_result import MeasurementResult


def test_sample_bitstrings():
    """Tests if the outcome is as expected with different noise models and
    error rate."""

    # testing default options i.e. noise model is amplitude damping with
    # the default error rate, shots is also the default number
    qc = cirq.Circuit(cirq.X(cirq.LineQubit(0))) + cirq.Circuit(
        cirq.measure(cirq.LineQubit(0))
    )
    result_default = sample_bitstrings(qc)
    assert result_default.nqubits == 1
    assert result_default.qubit_indices == (0,)
    assert result_default.shots == 8192
    assert isinstance(result_default, MeasurementResult)

    # test for sum(noise_level) = 0
    result_no_noise = sample_bitstrings(qc, noise_level=(0.00,), shots=10)
    assert np.allclose(
        result_no_noise.asarray,
        np.array([[1], [1], [1], [1], [1], [1], [1], [1], [1], [1]]),
    )

    # test result with inputs different from default
    result_not_default = sample_bitstrings(
        qc,
        cirq.GeneralizedAmplitudeDampingChannel,
        (0.1, 0.1),
        cirq.DensityMatrixSimulator(),
        8000,
    )
    assert result_not_default.nqubits == 1
    assert result_not_default.qubit_indices == (0,)
    assert result_not_default.shots == 8000


def test_compute_density_matrix():
    """Tests if the density matrix of a noisy circuit is output as expected."""

    qc = cirq.Circuit(cirq.X(cirq.LineQubit(0)))
    assert np.isclose(np.trace(compute_density_matrix(qc)), 1)
    assert np.isclose(
        np.trace(
            compute_density_matrix(
                qc, cirq.GeneralizedAmplitudeDampingChannel, (0.1, 0.1)
            )
        ),
        1,
    )


def test_execute_with_depolarizing_noise():
    """Tests if executor function for Cirq returns a proper expectation
    value when used for noisy depoalrizing simulation."""

    qc = cirq.Circuit()
    for _ in range(100):
        qc += cirq.X(cirq.LineQubit(0))

    observable = np.diag([0, 1])
    # Test 1
    noise1 = 0.0
    observable_exp_value = execute_with_depolarizing_noise(
        qc, observable, noise1
    )
    assert 0.0 == observable_exp_value

    # Test 2
    noise2 = 0.5
    observable_exp_value = execute_with_depolarizing_noise(
        qc, observable, noise2
    )
    assert np.isclose(observable_exp_value, 0.5)

    # Test 3
    noise3 = 0.001
    observable_exp_value = execute_with_depolarizing_noise(
        qc, observable, noise3
    )
    assert np.isclose(observable_exp_value, 0.062452)
