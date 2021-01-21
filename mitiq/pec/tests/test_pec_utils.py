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

from pytest import mark, raises
import numpy as np
from cirq import Gate, LineQubit, X, Y, Z, I, CNOT, depolarize, Circuit
from mitiq.pec.utils import (
    _simple_pauli_deco_dict,
    _max_ent_state_circuit,
    _operation_to_choi,
    _circuit_to_choi,
)

BASE_NOISE = 0.01
DECO_DICT = _simple_pauli_deco_dict(BASE_NOISE)
DECO_DICT_SIMP = _simple_pauli_deco_dict(BASE_NOISE, simplify_paulis=True)
NOISELESS_DECO_DICT = _simple_pauli_deco_dict(0)


def test_simple_pauli_deco_dict_cnot():
    """Tests that the _simple_pauli_deco_dict function returns a decomposition
    dicitonary which is consistent with a local depolarizing noise model.

    The channel acting on the state each qubit is assumed to be:
    D(rho) = = (1 - epsilon) rho + epsilon I/2
    = (1 - p) rho + p/3 (X rho X + Y rho Y^dag + Z rho Z)
    """

    # Deduce epsilon from BASE_NOISE
    epsilon = BASE_NOISE * 4.0 / 3.0
    c_neg = -(1 / 4) * epsilon / (1 - epsilon)
    c_pos = 1 - 3 * c_neg
    qreg = LineQubit.range(2)

    # Get the decomposition of a CNOT gate
    deco = DECO_DICT[CNOT.on(*qreg)]

    # The first term of 'deco' corresponds to no error occurring
    first_coefficient, first_imp_seq = deco[0]
    assert np.isclose(c_pos * c_pos, first_coefficient)
    assert first_imp_seq == [CNOT.on(*qreg)]
    # The second term corresponds to a Pauli X error on one qubit
    second_coefficient, second_imp_seq = deco[1]
    assert np.isclose(c_pos * c_neg, second_coefficient)
    assert second_imp_seq == [CNOT.on(*qreg), X.on(qreg[0])]
    # The last term corresponds to two Pauli Z errors on both qubits
    last_coefficient, last_imp_seq = deco[-1]
    assert np.isclose(c_neg * c_neg, last_coefficient)
    assert last_imp_seq == [CNOT.on(*qreg), Z.on(qreg[0]), Z.on(qreg[1])]


@mark.parametrize("gate", [X, Y, Z])
def test_simple_pauli_deco_dict_single_qubit(gate: Gate):
    """Tests that the _simple_pauli_deco_dict function returns a decomposition
    dicitonary which is consistent with a local depolarizing noise model.

    This is similar to test_simple_pauli_deco_dict_CNOT but applied to
    single-qubit gates.
    """
    epsilon = BASE_NOISE * 4.0 / 3.0
    c_neg = -(1 / 4) * epsilon / (1 - epsilon)
    c_pos = 1 - 3 * c_neg
    qreg = LineQubit.range(2)
    for q in qreg:
        deco = DECO_DICT[gate.on(q)]
        first_coefficient, first_imp_seq = deco[0]
        assert np.isclose(c_pos, first_coefficient)
        assert first_imp_seq == [gate.on(q)]
        second_coefficient, second_imp_seq = deco[1]
        assert np.isclose(c_neg, second_coefficient)
        assert second_imp_seq == [gate.on(q), X.on(q)]


@mark.parametrize("gate", [X, Y, Z])
def test_simplify_paulis_in_simple_pauli_deco_dict(gate: Gate):
    """Tests DECO_DICT_SIMP which is initialized using the 'simplify_paulis'
    option. This should produce decomposition dictionary in which Pauli
    sequences are simplified to single Pauli gates.
    """
    qreg = LineQubit.range(2)
    decomposition_dict = DECO_DICT_SIMP
    for q in qreg:
        deco = decomposition_dict[gate.on(q)]
        _, first_imp_seq = deco[0]
        assert first_imp_seq == [gate.on(q)]
        _, second_imp_seq = deco[1]
        input_times_x = {X: I, Y: Z, Z: Y}
        assert second_imp_seq == [input_times_x[gate].on(q)]


@mark.parametrize("gate", [X, Y, Z, CNOT])
def test_simple_pauli_deco_dict_with_choi(gate: Gate):
    """Tests the decomposition by comparing the exact Choi matrices."""
    qreg = LineQubit.range(gate.num_qubits())
    ideal_choi = _operation_to_choi(gate.on(*qreg))
    op_decomp = DECO_DICT[gate.on(*qreg)]
    choi_components = []
    for coeff, imp_seq in op_decomp:
        # Apply noise after each sequence.
        # NOTE: noise is not applied after each operation.
        noisy_sequence = [imp_seq] + [depolarize(BASE_NOISE)(q) for q in qreg]
        sequence_choi = _operation_to_choi(noisy_sequence)
        choi_components.append(coeff * sequence_choi)
    combination_choi = np.sum(choi_components, axis=0)
    assert np.allclose(ideal_choi, combination_choi)


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
