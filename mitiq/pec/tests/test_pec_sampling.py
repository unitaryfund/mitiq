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

"""Tests for mitiq.pec.sampling functions."""

import numpy as np
import pytest

import cirq
from cirq import (
    Circuit,
    Gate,
    LineQubit,
    NamedQubit,
    ops,
    depolarize,
    measure,
    measure_each,
)

from pyquil import (
    Program,
    gates,
)

from qiskit import QuantumRegister, QuantumCircuit

from mitiq.pec import (
    sample_sequence,
    sample_circuit,
    NoisyOperation,
    OperationRepresentation,
)
from mitiq.utils import _equal
from mitiq.pec.representations import (
    represent_operation_with_local_depolarizing_noise,
    represent_operations_in_circuit_with_local_depolarizing_noise,
)
from mitiq.pec.channels import _operation_to_choi, _circuit_to_choi


xcirq = Circuit(cirq.X(cirq.LineQubit(0)))
zcirq = Circuit(cirq.Z(cirq.LineQubit(0)))
cnotcirq = Circuit(cirq.CNOT(cirq.LineQubit(0), cirq.LineQubit(1)))
czcirq = Circuit(cirq.CZ(cirq.LineQubit(0), cirq.LineQubit(1)))


def test_sample_sequence_cirq():
    circuit = Circuit(cirq.H(LineQubit(0)))

    circuit.append(measure(LineQubit(0)))

    rep = OperationRepresentation(
        ideal=circuit,
        basis_expansion={
            NoisyOperation(xcirq): 0.5,
            NoisyOperation(zcirq): -0.5,
        },
    )

    for _ in range(5):
        seqs, signs, norm = sample_sequence(circuit, representations=[rep])
        assert isinstance(seqs[0], Circuit)
        assert signs[0] in {1, -1}
        assert norm == 1.0


def test_sample_sequence_qiskit():
    qreg = QuantumRegister(1)
    circuit = QuantumCircuit(qreg)
    _ = circuit.h(qreg)

    xcircuit = QuantumCircuit(qreg)
    _ = xcircuit.x(qreg)

    zcircuit = QuantumCircuit(qreg)
    _ = zcircuit.z(qreg)

    noisy_xop = NoisyOperation(xcircuit)
    noisy_zop = NoisyOperation(zcircuit)

    rep = OperationRepresentation(
        ideal=circuit,
        basis_expansion={noisy_xop: 0.5, noisy_zop: -0.5},
    )

    for _ in range(5):
        seqs, signs, norm = sample_sequence(circuit, representations=[rep])
        assert isinstance(seqs[0], QuantumCircuit)
        assert signs[0] in {1, -1}
        assert norm == 1.0


def test_sample_sequence_pyquil():
    circuit = Program(gates.H(0))

    noisy_xop = NoisyOperation(Program(gates.X(0)))
    noisy_zop = NoisyOperation(Program(gates.Z(0)))

    rep = OperationRepresentation(
        ideal=circuit,
        basis_expansion={noisy_xop: 0.5, noisy_zop: -0.5},
    )

    for _ in range(50):
        seqs, signs, norm = sample_sequence(circuit, representations=[rep])
        assert isinstance(seqs[0], Program)
        assert signs[0] in {1, -1}
        assert norm == 1.0


@pytest.mark.parametrize("seed", (1, 2, 3, 5))
def test_sample_sequence_cirq_random_state(seed):
    circuit = Circuit(cirq.H.on(LineQubit(0)))
    rep = OperationRepresentation(
        ideal=circuit,
        basis_expansion={
            NoisyOperation(xcirq): 0.5,
            NoisyOperation(zcirq): -0.5,
        },
    )

    sequences, signs, norm = sample_sequence(
        circuit, [rep], random_state=np.random.RandomState(seed)
    )

    for _ in range(20):
        new_sequences, new_signs, new_norm = sample_sequence(
            circuit, [rep], random_state=np.random.RandomState(seed)
        )
        assert _equal(new_sequences[0], sequences[0])
        assert new_signs[0] == signs[0]
        assert np.isclose(new_norm, norm)


@pytest.mark.parametrize("qubit", [0, 1])
def test_qubit_independent_representation_cirq(qubit):
    circuit = Circuit(cirq.H.on(LineQubit(0)), cirq.H.on(LineQubit(1)))
    circuit.append(measure_each(*LineQubit.range(2)))

    rep = OperationRepresentation(
        ideal=Circuit(cirq.H.on(LineQubit(qubit))),
        basis_expansion={
            NoisyOperation(xcirq): 0.5,
            NoisyOperation(zcirq): -0.5,
        },
        is_qubit_dependent=False,
    )

    for _ in range(5):
        seqs, signs, norm = sample_sequence(circuit, representations=[rep])
        assert isinstance(seqs[0], Circuit)
        assert signs[0] in {1, -1}
        assert norm == 1.0


@pytest.mark.parametrize("qubit", [0, 1])
def test_qubit_independent_representation_qiskit(qubit):

    qreg = QuantumRegister(2)

    circuit = QuantumCircuit(qreg)
    _ = circuit.h(qreg)

    xcircuit = QuantumCircuit(qreg)
    _ = xcircuit.x(qreg)

    zcircuit = QuantumCircuit(qreg)
    _ = zcircuit.z(qreg)

    noisy_xop = NoisyOperation(xcircuit)
    noisy_zop = NoisyOperation(zcircuit)

    ideal_op = QuantumCircuit(QuantumRegister(2))
    _ = ideal_op.h(qubit)
    rep = OperationRepresentation(
        ideal=ideal_op,
        basis_expansion={noisy_xop: 0.5, noisy_zop: -0.5},
        is_qubit_dependent=False,
    )

    for _ in range(5):
        seqs, signs, norm = sample_sequence(circuit, representations=[rep])
        assert isinstance(seqs[0], QuantumCircuit)
        assert signs[0] in {1, -1}
        assert norm == 1.0


@pytest.mark.parametrize("qubit", [0, 1])
def test_qubit_independent_representation_pyquil(qubit):
    circuit = Program(gates.H(0), gates.H(1))

    noisy_xop = NoisyOperation(Program(gates.X(0)))
    noisy_zop = NoisyOperation(Program(gates.Z(0)))

    rep = OperationRepresentation(
        ideal=circuit,
        basis_expansion={noisy_xop: 0.5, noisy_zop: -0.5},
        is_qubit_dependent=False,
    )

    for _ in range(50):
        seqs, signs, norm = sample_sequence(circuit, representations=[rep])
        assert isinstance(seqs[0], Program)
        assert signs[0] in {1, -1}
        assert norm == 1.0


@pytest.mark.parametrize(
    "reps",
    [
        [],
        [
            OperationRepresentation(
                Circuit(cirq.H.on(LineQubit(0))),
                {
                    NoisyOperation(xcirq): 0.5,
                    NoisyOperation(zcirq): -0.5,
                },
            )
        ],
    ],
)
def test_sample_sequence_no_representation(reps):
    circuit = Circuit(cirq.H.on(LineQubit(0)), cirq.H.on(LineQubit(1)))
    circuit.append(measure_each(*LineQubit.range(2)))
    with pytest.warns(UserWarning, match="No representation found for"):
        sequences, signs, norm = sample_sequence(circuit, reps)
        assert _equal(sequences[0], circuit, require_qubit_equality=True)
        assert signs == [1]
        assert norm == 1


@pytest.mark.parametrize("measure", [True, False])
def test_sample_circuit_cirq(measure):
    circuit = Circuit(
        ops.H.on(LineQubit(0)),
        ops.CNOT.on(*LineQubit.range(2)),
    )
    if measure:
        circuit.append(measure_each(*LineQubit.range(2)))

    h_rep = OperationRepresentation(
        ideal=circuit[:1],
        basis_expansion={
            NoisyOperation(xcirq): 0.6,
            NoisyOperation(zcirq): -0.6,
        },
    )

    cnot_rep = OperationRepresentation(
        ideal=circuit[1:2],
        basis_expansion={
            NoisyOperation(cnotcirq): 0.7,
            NoisyOperation(czcirq): -0.7,
        },
    )

    for _ in range(50):
        sampled_circuits, signs, norm = sample_circuit(
            circuit, representations=[h_rep, cnot_rep]
        )

        assert isinstance(sampled_circuits[0], Circuit)
        assert len(sampled_circuits[0]) == 2
        assert signs[0] in (-1, 1)
        assert norm >= 1


def test_sample_circuit_partial_representations():
    circuit = Circuit(
        ops.H.on(LineQubit(0)),
        ops.CNOT.on(*LineQubit.range(2)),
    )

    cnot_rep = OperationRepresentation(
        ideal=circuit[1:2],
        basis_expansion={
            NoisyOperation(cnotcirq): 0.7,
            NoisyOperation(czcirq): -0.7,
        },
    )

    for _ in range(10):
        with pytest.warns(UserWarning, match="No representation found for"):
            sampled_circuits, signs, norm = sample_circuit(
                circuit, representations=[cnot_rep]
            )
            assert isinstance(sampled_circuits[0], Circuit)
            assert len(sampled_circuits[0]) == 2
            assert signs[0] in (-1, 1)
            assert norm >= 1


def test_sample_circuit_pyquil():
    circuit = Program(gates.H(0), gates.CNOT(0, 1))

    h_rep = OperationRepresentation(
        ideal=circuit[:1],
        basis_expansion={
            NoisyOperation(Program(gates.X(0))): 0.6,
            NoisyOperation(Program(gates.Z(0))): -0.6,
        },
    )

    cnot_rep = OperationRepresentation(
        ideal=circuit[1:],
        basis_expansion={
            NoisyOperation(Program(gates.CNOT(0, 1))): 0.7,
            NoisyOperation(Program(gates.CZ(0, 1))): -0.7,
        },
    )

    for _ in range(50):
        sampled_circuits, signs, norm = sample_circuit(
            circuit, representations=[h_rep, cnot_rep]
        )

        assert isinstance(sampled_circuits[0], Program)
        assert len(sampled_circuits[0]) == 2
        assert signs[0] in (-1, 1)
        assert norm >= 1


def test_sample_circuit_with_seed():
    circ = Circuit([cirq.X.on(LineQubit(0)) for _ in range(10)])
    rep = OperationRepresentation(
        ideal=Circuit(cirq.X.on(LineQubit(0))),
        basis_expansion={
            NoisyOperation(zcirq): 1.0,
            NoisyOperation(xcirq): -1.0,
        },
    )

    expected_circuits, expected_signs, expected_norm = sample_circuit(
        circ, [rep], random_state=4
    )

    # Check we're not sampling the same operation every call to sample_sequence
    assert len(set(expected_circuits[0].all_operations())) > 1

    for _ in range(10):
        sampled_circuits, sampled_signs, sampled_norm = sample_circuit(
            circ, [rep], random_state=4
        )
        assert _equal(sampled_circuits[0], expected_circuits[0])
        assert sampled_signs[0] == expected_signs[0]
        assert sampled_norm == expected_norm


def test_sample_circuit_trivial_decomposition():
    circuit = Circuit(ops.H.on(NamedQubit("Q")))
    rep = OperationRepresentation(
        ideal=circuit, basis_expansion={NoisyOperation(circuit): 1.0}
    )

    sampled_circuits, signs, norm = sample_circuit(
        circuit, [rep], random_state=1
    )
    assert _equal(sampled_circuits[0], circuit)
    assert signs[0] == 1
    assert np.isclose(norm, 1)


BASE_NOISE = 0.02
qreg = LineQubit.range(2)


@pytest.mark.parametrize("gate", [cirq.Y, cirq.CNOT])
def test_sample_sequence_choi(gate: Gate):
    """Tests the sample_sequence by comparing the exact Choi matrices."""
    qreg = LineQubit.range(gate.num_qubits())
    ideal_op = gate.on(*qreg)
    ideal_circ = Circuit(ideal_op)
    noisy_op_tree = [ideal_op] + [depolarize(BASE_NOISE)(q) for q in qreg]

    ideal_choi = _operation_to_choi(ideal_op)
    noisy_choi = _operation_to_choi(noisy_op_tree)

    representation = represent_operation_with_local_depolarizing_noise(
        ideal_circ,
        BASE_NOISE,
    )

    choi_unbiased_estimates = []
    rng = np.random.RandomState(1)
    for _ in range(500):
        imp_seqs, signs, norm = sample_sequence(
            ideal_circ, [representation], random_state=rng
        )
        noisy_sequence = imp_seqs[0].with_noise(depolarize(BASE_NOISE))
        sequence_choi = _circuit_to_choi(noisy_sequence)
        choi_unbiased_estimates.append(norm * signs[0] * sequence_choi)

    choi_pec_estimate = np.average(choi_unbiased_estimates, axis=0)
    noise_error = np.linalg.norm(ideal_choi - noisy_choi)
    pec_error = np.linalg.norm(ideal_choi - choi_pec_estimate)

    assert pec_error < noise_error
    assert np.allclose(ideal_choi, choi_pec_estimate, atol=0.05)


def test_sample_circuit_choi():
    """Tests the sample_circuit by comparing the exact Choi matrices."""
    # A simple 2-qubit circuit
    qreg = LineQubit.range(2)
    ideal_circ = Circuit(
        cirq.X.on(qreg[0]),
        cirq.I.on(qreg[1]),
        cirq.CNOT.on(*qreg),
    )

    noisy_circuit = ideal_circ.with_noise(depolarize(BASE_NOISE))

    ideal_choi = _circuit_to_choi(ideal_circ)
    noisy_choi = _operation_to_choi(noisy_circuit)

    rep_list = represent_operations_in_circuit_with_local_depolarizing_noise(
        ideal_circuit=ideal_circ,
        noise_level=BASE_NOISE,
    )

    choi_unbiased_estimates = []
    rng = np.random.RandomState(1)
    for _ in range(500):
        imp_circs, signs, norm = sample_circuit(
            ideal_circ, rep_list, random_state=rng
        )
        noisy_imp_circ = imp_circs[0].with_noise(depolarize(BASE_NOISE))
        sequence_choi = _circuit_to_choi(noisy_imp_circ)
        choi_unbiased_estimates.append(norm * signs[0] * sequence_choi)

    choi_pec_estimate = np.average(choi_unbiased_estimates, axis=0)
    noise_error = np.linalg.norm(ideal_choi - noisy_choi)
    pec_error = np.linalg.norm(ideal_choi - choi_pec_estimate)

    assert pec_error < noise_error
    assert np.allclose(ideal_choi, choi_pec_estimate, atol=0.05)
