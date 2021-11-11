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

"""Unit tests for qiskit executors (qiskit_utils.py)."""
import numpy as np
from qiskit import QuantumCircuit

from mitiq.interface.mitiq_qiskit.qiskit_utils import (
    execute,
    execute_with_shots,
    execute_with_noise,
    execute_with_shots_and_noise,
    initialized_depolarizing_noise,
)

NOISE = 0.007
ONE_QUBIT_GS_PROJECTOR = np.array([[1, 0], [0, 0]])
TWO_QUBIT_GS_PROJECTOR = np.diag([1, 0, 0, 0])
SHOTS = 1_000


def test_execute():
    """ Tests the Qiskit wavefunction simulation executor returns
    appropriate expectation value given an observable.
    """

    circ = QuantumCircuit(1)
    expected_value = execute(circ, obs=ONE_QUBIT_GS_PROJECTOR)
    assert expected_value == 1.0

    second_circ = QuantumCircuit(1)
    second_circ.x(0)
    expected_value = execute(second_circ, obs=ONE_QUBIT_GS_PROJECTOR)
    assert expected_value == 0.0


def test_execute_with_shots():
    """ Tests the Qiskit wavefunction sampling simulation executor returns
    appropriate expectation value given an observable.
    """

    circ = QuantumCircuit(1, 1)
    expectation_value = execute_with_shots(
        circuit=circ, obs=ONE_QUBIT_GS_PROJECTOR, shots=SHOTS
    )
    assert expectation_value == 1.0

    second_circ = QuantumCircuit(1)
    second_circ.x(0)
    expectation_value = execute_with_shots(
        circuit=second_circ, obs=ONE_QUBIT_GS_PROJECTOR, shots=SHOTS
    )
    assert expectation_value == 0.0


def test_execute_with_depolarizing_noise_single_qubit():
    """ Tests the noisy sampling executor across increasing levels
    of single qubit gate noise
    """

    single_qubit_circ = QuantumCircuit(1)
    # noise model is defined on gates so include the gate to
    # demonstrate noise
    single_qubit_circ.z(0)

    noiseless_exp_value = 1.0

    expectation_value = execute_with_noise(
        circuit=single_qubit_circ,
        obs=ONE_QUBIT_GS_PROJECTOR,
        noise_model=initialized_depolarizing_noise(NOISE),
    )
    # anticipate that the expectation value will be less than
    # the noiseless simulation of the same circuit
    assert expectation_value < noiseless_exp_value


def test_execute_with_depolarizing_noise_two_qubit():
    """ Tests the noisy sampling executor across increasing levels of
    two qubit gate noise.
    """

    two_qubit_circ = QuantumCircuit(2)
    # noise model is defined on gates so include the gate to
    # demonstrate noise
    two_qubit_circ.cx(0, 1)

    noiseless_exp_value = 1.0

    expectation_value = execute_with_noise(
        circuit=two_qubit_circ,
        obs=TWO_QUBIT_GS_PROJECTOR,
        noise_model=initialized_depolarizing_noise(NOISE),
    )
    # anticipate that the expectation value will be less than
    # the noiseless simulation of the same circuit
    assert expectation_value < noiseless_exp_value


def test_execute_with_shots_and_depolarizing_noise_single_qubit():
    """ Tests the noisy sampling executor across increasing levels
    of single qubit gate noise.
    """

    single_qubit_circ = QuantumCircuit(1, 1)
    # noise model is defined on gates so include the gate to
    # demonstrate noise
    single_qubit_circ.z(0)

    noiseless_exp_value = 1.0

    expectation_value = execute_with_shots_and_noise(
        circuit=single_qubit_circ,
        obs=ONE_QUBIT_GS_PROJECTOR,
        noise_model=initialized_depolarizing_noise(NOISE),
        shots=SHOTS,
    )
    # anticipate that the expectation value will be less than
    # the noiseless simulation of the same circuit
    assert expectation_value < noiseless_exp_value


def test_execute_with_shots_and_depolarizing_noise_two_qubit():
    """ Tests the noisy sampling executor across increasing levels of
    two qubit gate noise.
    """

    two_qubit_circ = QuantumCircuit(2, 2)
    # noise model is defined on gates so include the gate to
    # demonstrate noise
    two_qubit_circ.cx(0, 1)

    noiseless_exp_value = 1.0

    expectation_value = execute_with_shots_and_noise(
        circuit=two_qubit_circ,
        obs=TWO_QUBIT_GS_PROJECTOR,
        noise_model=initialized_depolarizing_noise(NOISE),
        shots=SHOTS,
    )
    # anticipate that the expectation value will be less than
    # the noiseless simulation of the same circuit
    assert expectation_value < noiseless_exp_value


def test_circuit_is_not_mutated_by_executors():
    single_qubit_circ = QuantumCircuit(1, 1)
    single_qubit_circ.z(0)
    expected_circuit = single_qubit_circ.copy()
    execute_with_shots_and_noise(
        circuit=single_qubit_circ,
        obs=ONE_QUBIT_GS_PROJECTOR,
        noise_model=initialized_depolarizing_noise(NOISE),
        shots=SHOTS,
    )
    assert single_qubit_circ.data == expected_circuit.data
    assert single_qubit_circ == expected_circuit
    execute_with_noise(
        circuit=single_qubit_circ,
        obs=ONE_QUBIT_GS_PROJECTOR,
        noise_model=initialized_depolarizing_noise(NOISE),
    )
    assert single_qubit_circ.data == expected_circuit.data
    assert single_qubit_circ == expected_circuit
