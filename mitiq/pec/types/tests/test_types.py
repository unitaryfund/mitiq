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

import numpy as np
import pytest

import cirq
import pyquil
import qiskit

from mitiq.utils import _equal
from mitiq.pec.types import NoisyOperation, NoisyBasis


def test_init_with_gate():
    ideal_gate = cirq.Z
    real = np.zeros(shape=(4, 4))
    noisy_op = NoisyOperation.from_cirq(ideal_gate, real)
    assert isinstance(noisy_op._ideal, cirq.Circuit)

    assert _equal(
        noisy_op.ideal_circuit(),
        cirq.Circuit(ideal_gate.on(cirq.LineQubit(0))),
        require_qubit_equality=False,
    )
    assert noisy_op.qubits == (cirq.LineQubit(0),)
    assert np.allclose(noisy_op.ideal_matrix, cirq.unitary(cirq.Z))
    assert np.allclose(noisy_op.real_matrix, real)
    assert noisy_op.real_matrix is not real

    assert noisy_op._native_type == "cirq"
    assert _equal(noisy_op._native_ideal, noisy_op.ideal_circuit())


@pytest.mark.parametrize(
    "qubit",
    (cirq.LineQubit(0), cirq.GridQubit(1, 2), cirq.NamedQubit("Qubit")),
)
def test_init_with_operation(qubit):
    ideal_op = cirq.H.on(qubit)
    real = np.zeros(shape=(4, 4))
    noisy_op = NoisyOperation.from_cirq(ideal_op, real)

    assert isinstance(noisy_op._ideal, cirq.Circuit)
    assert _equal(
        noisy_op.ideal_circuit(),
        cirq.Circuit(ideal_op),
        require_qubit_equality=True,
    )
    assert noisy_op.qubits == (qubit,)
    assert np.allclose(noisy_op.ideal_matrix, cirq.unitary(ideal_op))
    assert np.allclose(noisy_op.real_matrix, real)
    assert noisy_op.real_matrix is not real

    assert noisy_op._native_type == "cirq"
    assert _equal(noisy_op._native_ideal, noisy_op.ideal_circuit())


def test_init_with_op_tree():
    qreg = cirq.LineQubit.range(2)
    ideal_ops = [cirq.H.on(qreg[0]), cirq.CNOT.on(*qreg)]
    real = np.zeros(shape=(16, 16))
    noisy_op = NoisyOperation.from_cirq(ideal_ops, real)

    assert isinstance(noisy_op._ideal, cirq.Circuit)
    assert _equal(
        noisy_op.ideal_circuit(),
        cirq.Circuit(ideal_ops),
        require_qubit_equality=True,
    )
    assert set(noisy_op.qubits) == set(qreg)
    assert np.allclose(
        noisy_op.ideal_matrix, cirq.unitary(cirq.Circuit(ideal_ops))
    )
    assert np.allclose(noisy_op.real_matrix, real)
    assert noisy_op.real_matrix is not real


def test_init_with_cirq_circuit():
    qreg = cirq.LineQubit.range(2)
    circ = cirq.Circuit(cirq.H.on(qreg[0]), cirq.CNOT.on(*qreg))
    real = np.zeros(shape=(16, 16))
    noisy_op = NoisyOperation(circ, real)

    assert isinstance(noisy_op._ideal, cirq.Circuit)
    assert _equal(noisy_op.ideal_circuit(), circ, require_qubit_equality=True)
    assert set(noisy_op.qubits) == set(qreg)
    assert np.allclose(noisy_op.ideal_matrix, cirq.unitary(circ))
    assert np.allclose(noisy_op.real_matrix, real)
    assert noisy_op.real_matrix is not real


def test_init_with_qiskit_circuit():
    qreg = qiskit.QuantumRegister(2)
    circ = qiskit.QuantumCircuit(qreg)
    _ = circ.h(qreg[0])
    _ = circ.cnot(*qreg)

    cirq_qreg = cirq.LineQubit.range(2)
    cirq_circ = cirq.Circuit(cirq.H.on(cirq_qreg[0]), cirq.CNOT.on(*cirq_qreg))

    real = np.zeros(shape=(16, 16))
    noisy_op = NoisyOperation(circ, real)
    assert isinstance(noisy_op._ideal, cirq.Circuit)
    assert _equal(noisy_op._ideal, cirq_circ)

    assert noisy_op.ideal_circuit() == circ
    assert noisy_op._native_ideal == circ
    assert noisy_op._native_type == "qiskit"

    assert np.allclose(noisy_op.ideal_matrix, cirq.unitary(cirq_circ))
    assert np.allclose(noisy_op.real_matrix, real)
    assert noisy_op.real_matrix is not real


@pytest.mark.parametrize(
    "gate",
    (
        cirq.H,
        cirq.H(cirq.LineQubit(0)),
        qiskit.extensions.HGate,
        qiskit.extensions.CHGate,
        pyquil.gates.H,
    ),
)
def test_init_with_gates_raises_error(gate):
    rng = np.random.RandomState(seed=1)
    with pytest.raises(TypeError, match="Arg `ideal` must be of type"):
        NoisyOperation(ideal=gate, real=rng.rand(4, 4))


def test_init_with_pyquil_program():
    circ = pyquil.Program(pyquil.gates.H(0), pyquil.gates.CNOT(0, 1))

    cirq_qreg = cirq.LineQubit.range(2)
    cirq_circ = cirq.Circuit(cirq.H.on(cirq_qreg[0]), cirq.CNOT.on(*cirq_qreg))

    real = np.zeros(shape=(16, 16))
    noisy_op = NoisyOperation(circ, real)
    assert isinstance(noisy_op._ideal, cirq.Circuit)
    assert _equal(noisy_op._ideal, cirq_circ)

    assert noisy_op.ideal_circuit() == circ
    assert noisy_op._native_ideal == circ
    assert noisy_op._native_type == "pyquil"

    assert np.allclose(noisy_op.ideal_matrix, cirq.unitary(cirq_circ))
    assert np.allclose(noisy_op.real_matrix, real)
    assert noisy_op.real_matrix is not real


def test_init_with_bad_types():
    ideal_ops = [cirq.H, cirq.CNOT]
    real = np.zeros(shape=(16, 16))
    with pytest.raises(ValueError, match="must be cirq.CIRCUIT_LIKE"):
        NoisyOperation.from_cirq(ideal_ops, real)


def test_init_dimension_mismatch_error():
    ideal = cirq.Circuit(cirq.H.on(cirq.LineQubit(0)))
    real = np.zeros(shape=(3, 3))
    with pytest.raises(ValueError, match="has shape"):
        NoisyOperation(ideal, real)


def test_add_simple():
    ideal = cirq.Circuit([cirq.X.on(cirq.NamedQubit("Q"))])
    real = np.random.rand(4, 4)

    noisy_op1 = NoisyOperation(ideal, real)
    noisy_op2 = NoisyOperation(ideal, real)

    noisy_op = noisy_op1 + noisy_op2

    correct = cirq.Circuit([cirq.X.on(cirq.NamedQubit("Q"))] * 2)

    assert _equal(noisy_op._ideal, correct, require_qubit_equality=True)
    assert np.allclose(noisy_op.ideal_matrix, np.identity(2))
    assert np.allclose(noisy_op.real_matrix, real @ real)


def test_add_bad_type():
    ideal = cirq.Circuit([cirq.X.on(cirq.NamedQubit("Q"))])
    real = np.random.rand(4, 4)

    noisy_op = NoisyOperation(ideal, real)

    with pytest.raises(ValueError, match="must be a NoisyOperation"):
        noisy_op + ideal


@pytest.mark.parametrize(
    "qreg",
    (
        cirq.LineQubit.range(5),
        cirq.GridQubit.square(5),
        [cirq.NamedQubit(str(i)) for i in range(5)],
    ),
)
def test_on_each_single_qubit(qreg):
    real = np.zeros(shape=(4, 4))
    noisy_ops = NoisyOperation.on_each(cirq.X, real, qubits=qreg)

    assert len(noisy_ops) == len(qreg)

    for i, op in enumerate(noisy_ops):
        assert np.allclose(op.real_matrix, real)
        assert op.num_qubits == 1
        assert list(op.ideal_circuit().all_qubits())[0] == qreg[i]


@pytest.mark.parametrize(
    "qubits",
    (
        [cirq.LineQubit.range(2), cirq.LineQubit.range(2, 4)],
        [cirq.LineQubit.range(5, 7), cirq.LineQubit.range(10, 12)],
        [cirq.GridQubit.rect(1, 2), cirq.GridQubit.rect(2, 1)],
    ),
)
def test_on_each_multiple_qubits(qubits):
    real_cnot = np.zeros(shape=(16, 16))
    noisy_ops = NoisyOperation.on_each(cirq.CNOT, real_cnot, qubits=qubits)
    assert len(noisy_ops) == 2

    for i, op in enumerate(noisy_ops):
        assert np.allclose(op.real_matrix, real_cnot)
        assert op.num_qubits == 2
        assert set(op.qubits) == set(qubits[i])


def test_on_each_multiple_qubits_bad_qubits_shape():
    real_cnot = np.zeros(shape=(16, 16))
    qubits = [cirq.LineQubit.range(3)]
    with pytest.raises(
        ValueError, match="Number of qubits in each register should be"
    ):
        NoisyOperation.on_each(cirq.CNOT, real_cnot, qubits=qubits)


@pytest.mark.parametrize(
    "qubit", (cirq.NamedQubit("New qubit"), cirq.GridQubit(2, 3))
)
def test_transform_qubits_single_qubit(qubit):
    real = np.zeros(shape=(4, 4))
    gate = cirq.H
    noisy_op = NoisyOperation.from_cirq(gate, real)

    assert set(noisy_op.qubits) != {qubit}
    noisy_op.transform_qubits(qubit)
    assert set(noisy_op.qubits) == {qubit}


@pytest.mark.parametrize(
    "qubits", (cirq.LineQubit.range(4, 6), cirq.GridQubit.rect(1, 2))
)
def test_transform_qubits_multiple_qubits(qubits):
    real = np.zeros(shape=(16, 16))
    qreg = [cirq.NamedQubit("Dummy 1"), cirq.NamedQubit("Dummy 2")]
    ideal = cirq.Circuit(cirq.ops.H.on(qreg[0]), cirq.ops.CNOT.on(*qreg))
    noisy_op = NoisyOperation(ideal, real)

    assert set(noisy_op.qubits) != set(qubits)
    assert np.allclose(noisy_op.real_matrix, real)

    noisy_op.transform_qubits(qubits)
    assert set(noisy_op.qubits) == set(qubits)
    assert np.allclose(noisy_op.real_matrix, real)


def test_with_qubits():
    real = np.zeros(shape=(16, 16))
    qreg = [cirq.NamedQubit("Dummy 1"), cirq.NamedQubit("Dummy 2")]
    ideal = cirq.Circuit(cirq.ops.H.on(qreg[0]), cirq.ops.CNOT.on(*qreg))
    noisy_op = NoisyOperation(ideal, real)

    assert set(noisy_op.qubits) == set(qreg)
    assert np.allclose(noisy_op.real_matrix, real)

    qubits = cirq.LineQubit.range(2)
    new_noisy_op = noisy_op.with_qubits(qubits)
    assert set(new_noisy_op.qubits) == set(qubits)
    assert np.allclose(new_noisy_op.real_matrix, real)


def test_extend_to_single_qubit():
    qbit, qreg = cirq.LineQubit(0), cirq.LineQubit.range(1, 10)
    ideal = cirq.Z.on(qbit)
    real = np.zeros(shape=(4, 4))
    noisy_op_on_one_qubit = NoisyOperation.from_cirq(ideal, real)

    noisy_ops_on_all_qubits = noisy_op_on_one_qubit.extend_to(qreg)

    assert isinstance(noisy_ops_on_all_qubits, list)
    assert len(noisy_ops_on_all_qubits) == 10

    for op in noisy_ops_on_all_qubits:
        assert _equal(op.ideal_circuit(), cirq.Circuit(ideal))
        assert np.allclose(op.ideal_matrix, cirq.unitary(ideal))
        assert np.allclose(op.real_matrix, real)


def test_noisy_basis_simple():
    rng = np.random.RandomState(seed=1)
    noisy_basis = NoisyBasis(
        NoisyOperation.from_cirq(ideal=cirq.I, real=rng.rand(4, 4)),
        NoisyOperation.from_cirq(ideal=cirq.X, real=rng.rand(4, 4)),
        NoisyOperation.from_cirq(ideal=cirq.Y, real=rng.rand(4, 4)),
        NoisyOperation.from_cirq(ideal=cirq.Z, real=rng.rand(4, 4)),
    )
    assert len(noisy_basis.elements) == 4
    assert noisy_basis.all_qubits() == {cirq.LineQubit(0)}


@pytest.mark.parametrize(
    "element",
    (
        cirq.X,
        cirq.CNOT(*cirq.LineQubit.range(2)),
        pyquil.gates.H,
        pyquil.gates.CNOT(0, 1),
        qiskit.extensions.HGate,
        qiskit.extensions.CnotGate,
    ),
)
def test_noisy_basis_bad_types(element):
    with pytest.raises(ValueError, match="must be of type `NoisyOperation`"):
        NoisyBasis(element)


def test_noisy_basis_add():
    rng = np.random.RandomState(seed=1)
    noisy_basis = NoisyBasis(
        NoisyOperation.from_cirq(ideal=cirq.I, real=rng.rand(4, 4)),
        NoisyOperation.from_cirq(ideal=cirq.X, real=rng.rand(4, 4)),
    )
    assert len(noisy_basis.elements) == 2

    noisy_basis.add(
        NoisyOperation.from_cirq(ideal=cirq.Y, real=rng.rand(4, 4)),
        NoisyOperation.from_cirq(ideal=cirq.Z, real=rng.rand(4, 4)),
    )
    assert len(noisy_basis.elements) == 4


def test_extend_to_simple():
    rng = np.random.RandomState(seed=1)
    noisy_basis = NoisyBasis(
        NoisyOperation.from_cirq(ideal=cirq.I, real=rng.rand(4, 4)),
        NoisyOperation.from_cirq(ideal=cirq.X, real=rng.rand(4, 4)),
    )
    assert len(noisy_basis.elements) == 2

    noisy_basis.extend_to(cirq.LineQubit.range(1, 3))
    assert len(noisy_basis.elements) == 6


def test_get_sequences_simple():
    rng = np.random.RandomState(seed=1)
    noisy_basis = NoisyBasis(
        NoisyOperation.from_cirq(ideal=cirq.I, real=rng.rand(4, 4)),
        NoisyOperation.from_cirq(ideal=cirq.X, real=rng.rand(4, 4)),
    )
    sequences = noisy_basis.get_sequences(length=2)
    assert all(isinstance(s, NoisyOperation) for s in sequences)

    q = cirq.LineQubit(0)
    expected_circuits = [
        cirq.Circuit(cirq.I(q), cirq.I(q)),
        cirq.Circuit(cirq.I(q), cirq.X(q)),
        cirq.Circuit(cirq.X(q), cirq.I(q)),
        cirq.Circuit(cirq.X(q), cirq.X(q)),
    ]
    for sequence, expected in zip(sequences, expected_circuits):
        assert _equal(
            sequence.ideal_circuit(return_type="cirq"),
            expected,
            require_qubit_equality=True,
        )
