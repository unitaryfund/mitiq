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
import qiskit
import numpy as np
from mitiq.mitiq_qiskit.qiskit_utils import (
    qs_wvf_sim,
    qs_wvf_sampling_sim,
    qs_noisy_sampling_sim,
)

observable = np.array([[1, 0], [0, 0]])
two_qubit_observable = np.diag([1, 0, 0, 0])
shots = 1000


def test_qs_wvf_sim():
    """ Tests the Qiskit waveform simulation executor returns
    appropriate expectation value given an observable
    """

    circ = qiskit.QuantumCircuit(qiskit.QuantumRegister(1))
    expected_value = qs_wvf_sim(circ=circ, obs=observable)
    assert 1.0 == expected_value

    circ.x(0)
    expected_value = qs_wvf_sim(circ=circ, obs=observable)
    assert 0.0 == expected_value


def test_qs_wvf_sampling_sim():
    """ Tests the Qiskit waveform sampling simulation executor returns
    appropriate expectation value given an observable
    """

    circ = qiskit.QuantumCircuit(qiskit.QuantumRegister(1))
    expectation_value = qs_wvf_sampling_sim(circ=circ, obs=observable,
                                            shots=shots)
    assert expectation_value == 1.0

    circ.x(0)
    expectation_value = qs_wvf_sampling_sim(circ=circ, obs=observable,
                                            shots=shots)
    assert expectation_value == 0.0


def test_qs_wvf_sampling_sim_error():
    """ Tests that an error is raised when a classical register
    is present in a Qiskit circuit for the "qs_wvf_sampling_sim"
    executor
    """

    circ = qiskit.QuantumCircuit(qiskit.QuantumRegister(1),
                                 qiskit.ClassicalRegister(1))
    with pytest.raises(ValueError, match="This executor only works on"
                       " programs with no classical bits."):
        qs_wvf_sampling_sim(circ=circ, obs=observable, shots=shots)


def test_qs_noisy_sampling_sim_single_qubit():
    """ Tests the noisy sampling executor and makes sure
    that the expectation value is less than 1 when single
    qubit gates are used.
    """

    single_qubit_circ = qiskit.QuantumCircuit(qiskit.QuantumRegister(1))
    single_qubit_circ.z(0)

    current_exp_value = 1.0

    for noise in [0.01, 0.1, 0.2, 1.0]:
        expectation_value = qs_noisy_sampling_sim(circ=single_qubit_circ,
                                                  obs=observable,
                                                  noise=noise,
                                                  shots=shots)
        # anticipate that the expectation value will be less than
        # the noiseless simulation of the same circuit
        assert expectation_value < current_exp_value
        current_exp_value = expectation_value


def test_qs_noisy_sampling_sim_two_qubit():
    """ Tests the noisy sampling executor and makes sure that
    the expectation value is less than 1 when two qubit gates
    are used.
    """

    two_qubit_circ = qiskit.QuantumCircuit(qiskit.QuantumRegister(2))
    two_qubit_circ.cx(0, 1)

    current_exp_value = 1.0

    for noise in [0.01, 0.1, 0.2, 1.0]:
        expectation_value = qs_noisy_sampling_sim(circ=two_qubit_circ,
                                                  obs=two_qubit_observable,
                                                  noise=noise,
                                                  shots=shots)
        # anticipate that the expectation value will be less than
        # the noiseless simulation of the same circuit
        assert expectation_value < current_exp_value
        current_exp_value = expectation_value


def test_qs_noisy_sampling_sim_error():
    """ Tests that an error is raised when a classical
    register is present in a Qiskit circuit for the
    "qs_noisy_sampling_sim" executor
    """
    circ = qiskit.QuantumCircuit(qiskit.QuantumRegister(1),
                                 qiskit.ClassicalRegister(1))
    with pytest.raises(ValueError, match="This executor only works on"
                       " programs with no classical bits."):
        qs_noisy_sampling_sim(circ=circ, obs=observable,
                              noise=0.01, shots=shots)
