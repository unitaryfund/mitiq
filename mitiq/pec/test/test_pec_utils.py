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

from pytest import mark
import numpy as np
from cirq import Gate, LineQubit, X, Y, Z, I, CNOT, depolarize
from mitiq.pec.utils import (
    _simple_pauli_deco_dict,
    get_coefficients,
    get_imp_sequences,
    get_one_norm,
    get_probabilities,
)
from mitiq.utils import _operation_to_choi

BASE_NOISE = 0.01
DECO_DICT = _simple_pauli_deco_dict(BASE_NOISE)
DECO_DICT_SIMP = _simple_pauli_deco_dict(BASE_NOISE, simplify_paulis=True)
NOISELESS_DECO_DICT = _simple_pauli_deco_dict(0)


def test_simple_pauli_deco_dict_CNOT():
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
    deco_dict = DECO_DICT_SIMP
    for q in qreg:
        deco = deco_dict[gate.on(q)]
        _, first_imp_seq = deco[0]
        assert first_imp_seq == [gate.on(q)]
        _, second_imp_seq = deco[1]
        input_times_x = {X: I, Y: Z, Z: Y}
        assert second_imp_seq == [input_times_x[gate].on(q)]


@mark.parametrize("gate", [X, Y, Z])
def test_get_coefficients(gate: Gate):
    q = LineQubit(0)
    coeffs = get_coefficients(gate.on(q), DECO_DICT)
    epsilon = BASE_NOISE * 4.0 / 3.0
    c_neg = -(1 / 4) * epsilon / (1 - epsilon)
    c_pos = 1 - 3 * c_neg
    assert np.isclose(np.sum(coeffs), 1.0)
    assert np.allclose(coeffs, [c_pos, c_neg, c_neg, c_neg])


def test_get_imp_sequences_with_simplify():
    deco_dict = DECO_DICT_SIMP
    q = LineQubit(0)
    expected_imp_sequences = [[X.on(q)], [I.on(q)], [Z.on(q)], [Y.on(q)]]
    assert get_imp_sequences(X.on(q), deco_dict) == expected_imp_sequences


@mark.parametrize("gate", [X, Y, Z])
def test_get_imp_sequences_no_simplify(gate: Gate):
    q = LineQubit(0)
    expected_imp_sequences = [
        [gate.on(q)],
        [gate.on(q), X.on(q)],
        [gate.on(q), Y.on(q)],
        [gate.on(q), Z.on(q)],
    ]
    assert get_imp_sequences(gate.on(q), DECO_DICT) == expected_imp_sequences


@mark.parametrize("gate", [X, Y, Z])
def test_get_one_norm(gate: Gate):
    q = LineQubit(0)
    epsilon = BASE_NOISE * 4.0 / 3.0
    expected_one_norm = (1.0 + 0.5 * epsilon) / (1.0 - epsilon)
    assert np.isclose(get_one_norm(gate.on(q), DECO_DICT), expected_one_norm)


@mark.parametrize("gate", [X, Y, Z])
def test_get_probabilities(gate: Gate):
    q = LineQubit(0)
    probs = get_probabilities(gate.on(q), DECO_DICT)
    assert all([p >= 0 for p in probs])
    assert np.isclose(sum(probs), 1.0)


@mark.parametrize("gate", [X, Y, Z, CNOT])
def test_simple_pauli_deco_dict_with_Choi(gate: Gate):
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
