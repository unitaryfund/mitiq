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

"""Tests related to mitiq.pec.utils functions."""

from pytest import raises
import numpy as np
from cirq import LineQubit, depolarize, Circuit
from mitiq.pec.utils import (
    _max_ent_state_circuit,
    _operation_to_choi,
    _circuit_to_choi,
)


def test_max_ent_state_circuit():
    """Tests 1-qubit and 2-qubit maximally entangled states are generated."""
    two_state = np.array([1, 0, 0, 1]) / np.sqrt(2)
    four_state = np.array(3 * [1, 0, 0, 0, 0] + [1]) / 2.0
    assert np.allclose(
        _max_ent_state_circuit(2).final_state_vector(), two_state
    )
    assert np.allclose(
        _max_ent_state_circuit(4).final_state_vector(), four_state
    )


def test_max_ent_state_circuit_error():
    """Tests an error is raised if the argument num_qubits is not valid."""
    for num_qubits in [0, 1, 3, 5, 2.0]:
        with raises(ValueError, match="The argument 'num_qubits' must"):
            _max_ent_state_circuit(num_qubits)
    # Test expected good arguments are ok
    for num_qubits in [2, 4, 6, 8]:
        assert _max_ent_state_circuit(num_qubits)


def test_operation_to_choi():
    """Tests the Choi matrix of a depolarizing channel is recovered."""
    # Define first the expected result
    base_noise = 0.01
    max_ent_state = np.array([1, 0, 0, 1]) / np.sqrt(2)
    identity_part = np.outer(max_ent_state, max_ent_state)
    mixed_part = np.eye(4) / 4.0
    epsilon = base_noise * 4.0 / 3.0
    choi = (1.0 - epsilon) * identity_part + epsilon * mixed_part
    # Choi matrix of the double application of a depolarizing channel
    choi_twice = sum(
        [
            ((1.0 - epsilon) ** 2 * identity_part),
            (2 * epsilon - epsilon ** 2) * mixed_part,
        ]
    )

    # Evaluate the Choi matrix of one or two depolarizing channels
    q = LineQubit(0)
    noisy_operation = depolarize(base_noise).on(q)
    noisy_sequence = [noisy_operation, noisy_operation]
    assert np.allclose(choi, _operation_to_choi(noisy_operation))
    assert np.allclose(choi_twice, _operation_to_choi(noisy_sequence))


def test_circuit_to_choi():
    """Tests _circuit_to_choi is consistent with _operation_to_choi."""
    base_noise = 0.01
    q = LineQubit(0)
    noisy_operation = depolarize(base_noise).on(q)
    assert np.allclose(
        _operation_to_choi(noisy_operation),
        _circuit_to_choi(Circuit(noisy_operation)),
    )
    noisy_sequence = [noisy_operation, noisy_operation]
    assert np.allclose(
        _operation_to_choi(noisy_sequence),
        _circuit_to_choi(Circuit(noisy_sequence)),
    )
