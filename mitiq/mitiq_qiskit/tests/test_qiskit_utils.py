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

"""Unit tests for conversions between Mitiq circuits and Qiskit circuits."""
import pytest
import sys
import qiskit
import numpy as np
from mitiq.utils import _equal
from mitiq.mitiq_qiskit.qiskit_utils import (
    qs_wvf_sim,
    qs_wvf_sampling_sim,
    qs_noisy_sampling_sim,
)

def test_qs_wvf_sim():
    """ Tests the Qiskit waveform simulation utility function.
    Tests 1-qubit waveforms are accurated simulated.
    """

    circ = qiskit.QuantumCircuit(qiskit.QuantumRegister(1))
    observable = np.array([[1,0], [0,0]])
    expected_value = qs_wvf_sim(circ, observable)
    assert 1.0 == expected_value
     
    circ = qiskit.QuantumCircuit(qiskit.QuantumRegister(1))
    circ.x(0)
    expected_value = qs_wvf_sim(circ, observable)
    assert 0.0 == expected_value

def test_qs_wvf_sampling_sim():

    shots = 100
    circ = qiskit.QuantumCircuit(qiskit.QuantumRegister(1))
    observable = np.array([[1,0], [0,0]])
    expectation_value = qs_wvf_sampling_sim(circ, observable, shots)
    assert expectation_value == 1.0

    circ = qiskit.QuantumCircuit(qiskit.QuantumRegister(1))
    circ.x(0)
    observable = np.array([[1,0], [0,0]])
    expectation_value = qs_wvf_sampling_sim(circ, observable, shots)
    assert expectation_value == 0.0

    circ = qiskit.QuantumCircuit(qiskit.QuantumRegister(1), qiskit.ClassicalRegister(1))
    with pytest.raises(ValueError, match="This executor only works on programs with no classical bits."):
        result = qs_wvf_sampling_sim(circ, observable, shots)


def test_qs_noisy_sampling_sim():

    shots = 1000
    n = 1 
    circ = qiskit.QuantumCircuit(qiskit.QuantumRegister(n))
    observable = np.array([[1,0], [0,0]])

    for noise in [0.01,0.1, 0.2, 1.0]:
        expectation_value = qs_noisy_sampling_sim(circ, observable, noise, shots)
        # anticipate that the expectation value will be less than the noiseless simulation
        # of the same circuit
        assert expectation_value < 1.0
