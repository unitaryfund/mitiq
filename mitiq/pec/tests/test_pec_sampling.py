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

import pytest
import numpy as np

from cirq import (
    Circuit,
    Gate,
    LineQubit,
    X,
    Y,
    CNOT,
    depolarize,
)

from mitiq.utils import _equal
from mitiq.pec.utils import (
    _simple_pauli_deco_dict,
    DecompositionDict,
    _operation_to_choi,
    _circuit_to_choi,
)
from mitiq.pec.sampling import sample_sequence, sample_circuit


import cirq
import pyquil
import qiskit

from mitiq.pec.sampling import _sample_sequence, _sample_circuit
from mitiq.pec.types import NoisyOperation, OperationDecomposition


def test_sample_sequence_cirq():
    circuit = cirq.Circuit(cirq.H(cirq.LineQubit(0)))

    noisy_xop = NoisyOperation.from_cirq(ideal=cirq.X)
    noisy_zop = NoisyOperation.from_cirq(ideal=cirq.Z)

    decomp = OperationDecomposition(
        ideal=circuit, basis_expansion={noisy_xop: 0.5, noisy_zop: -0.5,},
    )

    for _ in range(50):
        seq, sign, norm = _sample_sequence(circuit, decompositions=[decomp])
        assert isinstance(seq, cirq.Circuit)
        assert sign in {1, -1}
        assert norm == 1.0


def test_sample_sequence_qiskit():
    qreg = qiskit.QuantumRegister(1)
    circuit = qiskit.QuantumCircuit(qreg)
    _ = circuit.h(qreg)

    xcircuit = qiskit.QuantumCircuit(qreg)
    _ = xcircuit.x(qreg)

    zcircuit = qiskit.QuantumCircuit(qreg)
    _ = zcircuit.z(qreg)

    noisy_xop = NoisyOperation(xcircuit)
    noisy_zop = NoisyOperation(zcircuit)

    decomp = OperationDecomposition(
        ideal=circuit, basis_expansion={noisy_xop: 0.5, noisy_zop: -0.5,},
    )

    for _ in range(50):
        seq, sign, norm = _sample_sequence(circuit, decompositions=[decomp])
        assert isinstance(seq, qiskit.QuantumCircuit)
        assert sign in {1, -1}
        assert norm == 1.0


def test_sample_sequence_pyquil():
    circuit = pyquil.Program(pyquil.gates.H(0))

    noisy_xop = NoisyOperation(pyquil.Program(pyquil.gates.X(0)))
    noisy_zop = NoisyOperation(pyquil.Program(pyquil.gates.Z(0)))

    decomp = OperationDecomposition(
        ideal=circuit, basis_expansion={noisy_xop: 0.5, noisy_zop: -0.5,},
    )

    for _ in range(50):
        seq, sign, norm = _sample_sequence(circuit, decompositions=[decomp])
        assert isinstance(seq, pyquil.Program)
        assert sign in {1, -1}
        assert norm == 1.0


@pytest.mark.parametrize("seed", (1, 2, 3, 5))
def test_sample_sequence_cirq_random_state(seed):
    circuit = cirq.Circuit(cirq.H.on(cirq.LineQubit(0)))
    decomposition = OperationDecomposition(
        ideal=circuit,
        basis_expansion={
            NoisyOperation.from_cirq(ideal=cirq.X): 0.5,
            NoisyOperation.from_cirq(ideal=cirq.Z): -0.5,
        },
    )

    sequence, sign, norm = _sample_sequence(
        circuit, [decomposition], random_state=np.random.RandomState(seed)
    )

    for _ in range(20):
        new_sequence, new_sign, new_norm = _sample_sequence(
            circuit, [decomposition], random_state=np.random.RandomState(seed)
        )
        assert _equal(new_sequence, sequence)
        assert new_sign == sign
        assert np.isclose(new_norm, norm)


def test_sample_circuit_cirq():
    circuit = cirq.Circuit(
        cirq.ops.H.on(cirq.LineQubit(0)),
        cirq.ops.CNOT.on(*cirq.LineQubit.range(2)),
    )

    hdecomposition = OperationDecomposition(
        ideal=circuit[:1],
        basis_expansion={
            NoisyOperation.from_cirq(ideal=cirq.X): 0.6,
            NoisyOperation.from_cirq(ideal=cirq.Z): -0.6,
        },
    )

    cnot_decomposition = OperationDecomposition(
        ideal=circuit[1:],
        basis_expansion={
            NoisyOperation.from_cirq(ideal=cirq.CNOT): 0.7,
            NoisyOperation.from_cirq(ideal=cirq.CZ): -0.7,
        },
    )

    for _ in range(50):
        sampled_circuit, sign, norm = _sample_circuit(
            circuit, decompositions=[hdecomposition, cnot_decomposition]
        )

        assert isinstance(sampled_circuit, cirq.Circuit)
        assert len(sampled_circuit) == 2
        assert sign in (-1, 1)
        assert norm >= 1


def test_sample_circuit_pyquil():
    circuit = pyquil.Program(pyquil.gates.H(0), pyquil.gates.CNOT(0, 1))

    hdecomposition = OperationDecomposition(
        ideal=circuit[:1],
        basis_expansion={
            NoisyOperation(pyquil.Program(pyquil.gates.X(0))): 0.6,
            NoisyOperation(pyquil.Program(pyquil.gates.Z(0))): -0.6,
        },
    )

    cnot_decomposition = OperationDecomposition(
        ideal=circuit[1:],
        basis_expansion={
            NoisyOperation(pyquil.Program(pyquil.gates.CNOT(0, 1))): 0.7,
            NoisyOperation(pyquil.Program(pyquil.gates.CZ(0, 1))): -0.7,
        },
    )

    for _ in range(50):
        sampled_circuit, sign, norm = _sample_circuit(
            circuit, decompositions=[hdecomposition, cnot_decomposition]
        )

        assert isinstance(sampled_circuit, pyquil.Program)
        assert len(sampled_circuit) == 2
        assert sign in (-1, 1)
        assert norm >= 1


# Old tests.
BASE_NOISE = 0.02
DECO_DICT = _simple_pauli_deco_dict(BASE_NOISE)
DECO_DICT_SIMP = _simple_pauli_deco_dict(BASE_NOISE, simplify_paulis=True)
NOISELESS_DECO_DICT = _simple_pauli_deco_dict(0)

# Simple 2-qubit circuit
qreg = LineQubit.range(2)
twoq_circ = Circuit(X.on(qreg[0]), CNOT.on(*qreg),)


def test_sample_circuit_with_seed():
    decomp = _simple_pauli_deco_dict(0.7, simplify_paulis=True)
    circ = Circuit([X.on(LineQubit(0)) for _ in range(10)])

    expected = sample_circuit(circ, decomp, random_state=4)[0]

    # Check we're not sampling the same operation every call to sample_sequence
    assert len(set(expected.all_operations())) > 1

    for _ in range(10):
        sampled = sample_circuit(circ, decomp, random_state=4)[0]
        assert _equal(sampled, expected)


def test_sample_circuit_types_trivial():
    imp_circuit, sign, norm = sample_circuit(twoq_circ, NOISELESS_DECO_DICT)
    assert imp_circuit == twoq_circ
    assert sign == 1
    assert np.isclose(norm, 1)


@pytest.mark.parametrize("gate", [Y, CNOT])
def test_sample_sequence_choi(gate: Gate):
    """Tests the sample_sequence by comparing the exact Choi matrices."""
    qreg = LineQubit.range(gate.num_qubits())
    ideal_op = gate.on(*qreg)
    noisy_op_tree = [ideal_op] + [depolarize(BASE_NOISE)(q) for q in qreg]
    ideal_choi = _operation_to_choi(gate.on(*qreg))
    noisy_choi = _operation_to_choi(noisy_op_tree)
    choi_unbiased_estimates = []
    rng = np.random.RandomState(1)
    for _ in range(500):
        imp_seq, sign, norm = sample_sequence(
            gate.on(*qreg), DECO_DICT, random_state=rng
        )
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


@pytest.mark.parametrize("decomposition_dict", [DECO_DICT, DECO_DICT_SIMP])
def test_sample_circuit_choi(decomposition_dict: DecompositionDict):
    """Tests the sample_circuit by comparing the exact Choi matrices."""
    ideal_choi = _circuit_to_choi(twoq_circ)
    noisy_circuit = twoq_circ.with_noise(depolarize(BASE_NOISE))
    noisy_choi = _circuit_to_choi(noisy_circuit)
    choi_unbiased_estimates = []
    for _ in range(500):
        imp_circuit, sign, norm = sample_circuit(twoq_circ, decomposition_dict)
        noisy_imp_circuit = imp_circuit.with_noise(depolarize(BASE_NOISE))
        imp_circuit_choi = _circuit_to_choi(noisy_imp_circuit)
        choi_unbiased_estimates.append(norm * sign * imp_circuit_choi)
    choi_pec_estimate = np.average(choi_unbiased_estimates, axis=0)

    noise_error = np.linalg.norm(ideal_choi - noisy_choi)
    pec_error = np.linalg.norm(ideal_choi - choi_pec_estimate)
    assert pec_error < noise_error
    assert np.allclose(ideal_choi, choi_pec_estimate, atol=0.05)
