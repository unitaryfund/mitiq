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

"""Unit tests for qiskit executors (qiskit_utils.py)."""
import pytest
import numpy as np
from qiskit import QuantumCircuit

from mitiq.mitiq_qiskit.qiskit_utils import (
    execute,
    execute_with_shots,
    execute_with_depolarizing_noise,
    execute_with_shots_and_depolarizing_noise,
)

OBSERVABLE = np.array([[1, 0], [0, 0]])
TWO_QUBIT_OBSERVABLE = np.diag([1, 0, 0, 0])
SHOTS = 1000


def test_execute():
    """ Tests the Qiskit waveform simulation executor returns
    appropriate expectation value given an observable
    """

    circ = QuantumCircuit(1)
    expected_value = execute(circ, obs=OBSERVABLE)
    assert 1.0 == expected_value

    second_circ = QuantumCircuit(1)
    second_circ.x(0)
    expected_value = execute(second_circ, obs=OBSERVABLE)
    assert 0.0 == expected_value


def test_execute_with_shots():
    """ Tests the Qiskit waveform sampling simulation executor returns
    appropriate expectation value given an observable
    """

    circ = QuantumCircuit(1)
    expectation_value = execute_with_shots(
        circ=circ, obs=OBSERVABLE, shots=SHOTS
    )
    assert expectation_value == 1.0

    second_circ = QuantumCircuit(1)
    second_circ.x(0)
    expectation_value = execute_with_shots(
        circ=second_circ, obs=OBSERVABLE, shots=SHOTS
    )
    assert expectation_value == 0.0


def test_execute_with_shots_error():
    """ Tests that an error is raised when a classical register
    is present in a Qiskit circuit for the "execute_with_shots"
    executor
    """

    circ = QuantumCircuit(1, 1)
    with pytest.raises(
        ValueError,
        match="This executor only works on programs with no classical bits.",
    ):
        execute_with_shots(circ=circ, obs=OBSERVABLE, shots=SHOTS)


def test_execute_with_depolarizing_noise_single_qubit():
    """ Tests the noisy sampling executor across increasing levels
    of single qubit gate noise
    """

    single_qubit_circ = QuantumCircuit(1)
    single_qubit_circ.z(0)

    noiseless_exp_value = 1.0

    for noise in [0.01, 0.1, 0.2, 1.0]:
        expectation_value = execute_with_depolarizing_noise(
            circ=single_qubit_circ, obs=OBSERVABLE, noise=noise,
        )
        # anticipate that the expectation value will be less than
        # the noiseless simulation of the same circuit
        assert expectation_value < noiseless_exp_value


def test_execute_with_depolarizing_noise_two_qubit():
    """ Tests the noisy sampling executor across increasing levels of
    two qubit gate noise
    """

    two_qubit_circ = QuantumCircuit(2)
    two_qubit_circ.cx(0, 1)

    noiseless_exp_value = 1.0

    for noise in [0.01, 0.1, 0.2, 1.0]:
        expectation_value = execute_with_depolarizing_noise(
            circ=two_qubit_circ, obs=TWO_QUBIT_OBSERVABLE, noise=noise,
        )
        # anticipate that the expectation value will be less than
        # the noiseless simulation of the same circuit
        assert expectation_value < noiseless_exp_value


def test_execute_with_depolarizing_noise_error():
    """ Tests that an error is raised when a classical
    register is present in a Qiskit circuit for the
    "execute_with_depolarizing_noise" executor
    """
    circ = QuantumCircuit(1, 1)
    with pytest.raises(
        ValueError,
        match="This executor only works on programs with no classical bits.",
    ):
        execute_with_shots_and_depolarizing_noise(
            circ=circ, obs=OBSERVABLE, noise=0.01, shots=SHOTS
        )


def test_execute_with_shots_and_depolarizing_noise_single_qubit():
    """ Tests the noisy sampling executor across increasing levels
    of single qubit gate noise
    """

    single_qubit_circ = QuantumCircuit(1)
    single_qubit_circ.z(0)

    noiseless_exp_value = 1.0

    for noise in [0.01, 0.1, 0.2, 1.0]:
        expectation_value = execute_with_shots_and_depolarizing_noise(
            circ=single_qubit_circ, obs=OBSERVABLE, noise=noise, shots=SHOTS,
        )
        # anticipate that the expectation value will be less than
        # the noiseless simulation of the same circuit
        assert expectation_value < noiseless_exp_value


def test_execute_with_shots_and_depolarizing_noise_two_qubit():
    """ Tests the noisy sampling executor across increasing levels of
    two qubit gate noise
    """

    two_qubit_circ = QuantumCircuit(2)
    two_qubit_circ.cx(0, 1)

    noiseless_exp_value = 1.0

    for noise in [0.01, 0.1, 0.2, 1.0]:
        expectation_value = execute_with_shots_and_depolarizing_noise(
            circ=two_qubit_circ,
            obs=TWO_QUBIT_OBSERVABLE,
            noise=noise,
            shots=SHOTS,
        )
        # anticipate that the expectation value will be less than
        # the noiseless simulation of the same circuit
        assert expectation_value < noiseless_exp_value


def test_execute_with_shots_and_depolarizing_noise_error():
    """ Tests that an error is raised when a classical
    register is present in a Qiskit circuit for the
    "execute_with_shots_and_depolarizing_noise" executor
    """
    circ = QuantumCircuit(1, 1)
    with pytest.raises(
        ValueError,
        match="This executor only works on programs with no classical bits.",
    ):
        execute_with_shots_and_depolarizing_noise(
            circ=circ, obs=OBSERVABLE, noise=0.01, shots=SHOTS
        )
