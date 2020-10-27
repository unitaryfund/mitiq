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

"""Tests related to mitiq.pec.sampling functions."""

from pytest import mark
import numpy as np

from cirq import (
    Circuit,
    Operation,
    Gate,
    LineQubit,
    X,
    Y,
    Z,
    CNOT,
    depolarize,
)

from mitiq.pec.utils import _simple_pauli_deco_dict, DecoType
from mitiq.pec.sampling import sample_sequence, sample_circuit
from mitiq.utils import _operation_to_choi, _circuit_to_choi

BASE_NOISE = 0.02
DECO_DICT = _simple_pauli_deco_dict(BASE_NOISE)
DECO_DICT_SIMP = _simple_pauli_deco_dict(BASE_NOISE, simplify_paulis=True)
NOISELESS_DECO_DICT = _simple_pauli_deco_dict(0)

# Simple 2-qubit circuit
qreg = LineQubit.range(2)
twoq_circ = Circuit(
    X.on(qreg[0]),
    CNOT.on(*qreg),
)


@mark.parametrize("gate", [X, Y, Z, CNOT])
def test_sample_sequence_types(gate: Gate):
    num_qubits = gate.num_qubits()
    qreg = LineQubit.range(num_qubits)
    for _ in range(1000):
        imp_seq, sign, norm = sample_sequence(gate.on(*qreg), DECO_DICT)
        assert all([isinstance(op, Operation) for op in imp_seq])
        assert sign in {1, -1}
        assert norm > 1


def test_sample_circuit_types():
    imp_circuit, sign, norm = sample_circuit(twoq_circ, DECO_DICT)
    assert isinstance(imp_circuit, Circuit)
    assert sign in {1, -1}
    assert norm > 1


def test_sample_circuit_types_trivial():
    imp_circuit, sign, norm = sample_circuit(twoq_circ, NOISELESS_DECO_DICT)
    assert imp_circuit == twoq_circ
    assert sign == 1
    assert np.isclose(norm, 1)


@mark.parametrize("gate", [Y, CNOT])
def test_sample_sequence_choi(gate: Gate):
    """Tests the sample_sequence by comparing the exact Choi matrices."""
    qreg = LineQubit.range(gate.num_qubits())
    ideal_op = gate.on(*qreg)
    noisy_op_tree = [ideal_op] + [depolarize(BASE_NOISE)(q) for q in qreg]
    ideal_choi = _operation_to_choi(gate.on(*qreg))
    noisy_choi = _operation_to_choi(noisy_op_tree)
    choi_unbiased_estimates = []
    for _ in range(500):
        imp_seq, sign, norm = sample_sequence(gate.on(*qreg), DECO_DICT)
        # Apply noise after each sequence.
        # NOTE: noise is not applied after each operation.
        noisy_sequence = [imp_seq] + [depolarize(BASE_NOISE)(q) for q in qreg]
        sequence_choi = _operation_to_choi(noisy_sequence)
        choi_unbiased_estimates.append(norm * sign * sequence_choi)
    choi_pec_estimate = np.average(choi_unbiased_estimates, axis=0)
    noise_error = np.linalg.norm(ideal_choi - noisy_choi)
    pec_error = np.linalg.norm(ideal_choi - choi_pec_estimate)
    assert pec_error < noise_error
    assert np.allclose(ideal_choi, choi_pec_estimate, atol=0.05)


@mark.parametrize("deco_dict", [DECO_DICT, DECO_DICT_SIMP])
def test_sample_circuit_choi(deco_dict: DecoType):
    """Tests the sample_circuit by comparing the exact Choi matrices."""
    ideal_choi = _circuit_to_choi(twoq_circ)
    noisy_circuit = twoq_circ.with_noise(depolarize(BASE_NOISE))
    noisy_choi = _circuit_to_choi(noisy_circuit)
    choi_unbiased_estimates = []
    for _ in range(500):
        imp_circuit, sign, norm = sample_circuit(twoq_circ, deco_dict)
        noisy_imp_circuit = imp_circuit.with_noise(depolarize(BASE_NOISE))
        imp_circuit_choi = _circuit_to_choi(noisy_imp_circuit)
        choi_unbiased_estimates.append(norm * sign * imp_circuit_choi)
    choi_pec_estimate = np.average(choi_unbiased_estimates, axis=0)

    noise_error = np.linalg.norm(ideal_choi - noisy_choi)
    pec_error = np.linalg.norm(ideal_choi - choi_pec_estimate)
    assert pec_error < noise_error
    assert np.allclose(ideal_choi, choi_pec_estimate, atol=0.05)
