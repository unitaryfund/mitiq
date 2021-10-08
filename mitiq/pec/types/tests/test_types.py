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
from mitiq.pec.types import (
    NoisyBasis,
    NoisyOperation,
    OperationRepresentation,
)


def test_init_with_gate():
    ideal_gate = cirq.Z
    real = np.zeros(shape=(4, 4))
    noisy_op = NoisyOperation.from_cirq(ideal_gate, real)
    assert isinstance(noisy_op._circuit, cirq.Circuit)

    assert _equal(
        noisy_op.circuit(),
        cirq.Circuit(ideal_gate.on(cirq.LineQubit(0))),
        require_qubit_equality=False,
    )
    assert noisy_op.qubits == (cirq.LineQubit(0),)
    assert np.allclose(noisy_op.ideal_unitary, cirq.unitary(cirq.Z))
    assert np.allclose(noisy_op.channel_matrix, real)
    assert noisy_op.channel_matrix is not real

    assert noisy_op._native_type == "cirq"
    assert _equal(noisy_op._native_circuit, noisy_op.circuit())


@pytest.mark.parametrize(
    "qubit",
    (cirq.LineQubit(0), cirq.GridQubit(1, 2), cirq.NamedQubit("Qubit")),
)
def test_init_with_operation(qubit):
    ideal_op = cirq.H.on(qubit)
    real = np.zeros(shape=(4, 4))
    noisy_op = NoisyOperation.from_cirq(ideal_op, real)

    assert isinstance(noisy_op._circuit, cirq.Circuit)
    assert _equal(
        noisy_op.circuit(),
        cirq.Circuit(ideal_op),
        require_qubit_equality=True,
    )
    assert noisy_op.qubits == (qubit,)
    assert np.allclose(noisy_op.ideal_unitary, cirq.unitary(ideal_op))
    assert np.allclose(noisy_op.channel_matrix, real)
    assert noisy_op.channel_matrix is not real

    assert noisy_op._native_type == "cirq"
    assert _equal(noisy_op._native_circuit, noisy_op.circuit())


def test_init_with_op_tree():
    qreg = cirq.LineQubit.range(2)
    ideal_ops = [cirq.H.on(qreg[0]), cirq.CNOT.on(*qreg)]
    real = np.zeros(shape=(16, 16))
    noisy_op = NoisyOperation.from_cirq(ideal_ops, real)

    assert isinstance(noisy_op._circuit, cirq.Circuit)
    assert _equal(
        noisy_op.circuit(),
        cirq.Circuit(ideal_ops),
        require_qubit_equality=True,
    )
    assert set(noisy_op.qubits) == set(qreg)
    assert np.allclose(
        noisy_op.ideal_unitary, cirq.unitary(cirq.Circuit(ideal_ops))
    )
    assert np.allclose(noisy_op.channel_matrix, real)
    assert noisy_op.channel_matrix is not real


def test_init_with_cirq_circuit():
    qreg = cirq.LineQubit.range(2)
    circ = cirq.Circuit(cirq.H.on(qreg[0]), cirq.CNOT.on(*qreg))
    real = np.zeros(shape=(16, 16))
    noisy_op = NoisyOperation(circ, real)

    assert isinstance(noisy_op._circuit, cirq.Circuit)
    assert _equal(noisy_op.circuit(), circ, require_qubit_equality=True)
    assert set(noisy_op.qubits) == set(qreg)
    assert np.allclose(noisy_op.ideal_unitary, cirq.unitary(circ))
    assert np.allclose(noisy_op.channel_matrix, real)
    assert noisy_op.channel_matrix is not real


def test_init_with_qiskit_circuit():
    qreg = qiskit.QuantumRegister(2)
    circ = qiskit.QuantumCircuit(qreg)
    _ = circ.h(qreg[0])
    _ = circ.cnot(*qreg)

    cirq_qreg = cirq.LineQubit.range(2)
    cirq_circ = cirq.Circuit(cirq.H.on(cirq_qreg[0]), cirq.CNOT.on(*cirq_qreg))

    real = np.zeros(shape=(16, 16))
    noisy_op = NoisyOperation(circ, real)
    assert isinstance(noisy_op._circuit, cirq.Circuit)
    assert _equal(noisy_op._circuit, cirq_circ)

    assert noisy_op.circuit() == circ
    assert noisy_op._native_circuit == circ
    assert noisy_op._native_type == "qiskit"

    assert np.allclose(noisy_op.ideal_unitary, cirq.unitary(cirq_circ))
    assert np.allclose(noisy_op.channel_matrix, real)
    assert noisy_op.channel_matrix is not real


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
    with pytest.raises(TypeError, match="Arg `circuit` must be one of"):
        NoisyOperation(circuit=gate, channel_matrix=rng.rand(4, 4))


def test_init_with_pyquil_program():
    circ = pyquil.Program(pyquil.gates.H(0), pyquil.gates.CNOT(0, 1))

    cirq_qreg = cirq.LineQubit.range(2)
    cirq_circ = cirq.Circuit(cirq.H.on(cirq_qreg[0]), cirq.CNOT.on(*cirq_qreg))

    real = np.zeros(shape=(16, 16))
    noisy_op = NoisyOperation(circ, real)
    assert isinstance(noisy_op._circuit, cirq.Circuit)
    assert _equal(noisy_op._circuit, cirq_circ)

    assert noisy_op.circuit() == circ
    assert noisy_op._native_circuit == circ
    assert noisy_op._native_type == "pyquil"

    assert np.allclose(noisy_op.ideal_unitary, cirq.unitary(cirq_circ))
    assert np.allclose(noisy_op.channel_matrix, real)
    assert noisy_op.channel_matrix is not real


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


def test_unknown_channel_matrix():
    qreg = qiskit.QuantumRegister(2)
    circ = qiskit.QuantumCircuit(qreg)
    _ = circ.h(qreg[0])
    _ = circ.cnot(*qreg)

    cirq_qreg = cirq.LineQubit.range(2)
    cirq_circ = cirq.Circuit(cirq.H.on(cirq_qreg[0]), cirq.CNOT.on(*cirq_qreg))

    noisy_op = NoisyOperation(circ)
    assert isinstance(noisy_op._circuit, cirq.Circuit)
    assert _equal(noisy_op._circuit, cirq_circ)

    assert noisy_op.circuit() == circ
    assert noisy_op._native_circuit == circ
    assert noisy_op._native_type == "qiskit"

    assert np.allclose(noisy_op.ideal_unitary, cirq.unitary(cirq_circ))

    with pytest.raises(ValueError, match="The channel matrix is unknown."):
        _ = noisy_op.channel_matrix


def test_add_simple():
    circuit1 = cirq.Circuit([cirq.X.on(cirq.NamedQubit("Q"))])
    circuit2 = cirq.Circuit([cirq.Y.on(cirq.NamedQubit("Q"))])

    super_op1 = np.random.rand(4, 4)
    super_op2 = np.random.rand(4, 4)

    noisy_op1 = NoisyOperation(circuit1, super_op1)
    noisy_op2 = NoisyOperation(circuit2, super_op2)

    noisy_op = noisy_op1 + noisy_op2

    correct = cirq.Circuit(
        [cirq.X.on(cirq.NamedQubit("Q")), cirq.Y.on(cirq.NamedQubit("Q"))],
    )

    assert _equal(noisy_op._circuit, correct, require_qubit_equality=True)
    assert np.allclose(noisy_op.channel_matrix, super_op2 @ super_op1)


def test_add_pyquil_noisy_operations():
    ideal = pyquil.Program(pyquil.gates.X(0))
    real = np.random.rand(4, 4)

    noisy_op1 = NoisyOperation(ideal, real)
    noisy_op2 = NoisyOperation(ideal, real)

    noisy_op = noisy_op1 + noisy_op2

    correct = cirq.Circuit([cirq.X.on(cirq.NamedQubit("Q"))] * 2)

    assert _equal(noisy_op._circuit, correct, require_qubit_equality=False)
    assert np.allclose(noisy_op.ideal_unitary, np.identity(2))
    assert np.allclose(noisy_op.channel_matrix, real @ real)


def test_add_qiskit_noisy_operations():
    qreg = qiskit.QuantumRegister(1)
    ideal = qiskit.QuantumCircuit(qreg)
    _ = ideal.x(qreg)
    real = np.random.rand(4, 4)

    noisy_op1 = NoisyOperation(ideal, real)
    noisy_op2 = NoisyOperation(ideal, real)

    noisy_op = noisy_op1 + noisy_op2

    correct = cirq.Circuit([cirq.X.on(cirq.NamedQubit("Q"))] * 2)

    assert _equal(noisy_op._circuit, correct, require_qubit_equality=False)
    assert np.allclose(noisy_op.ideal_unitary, np.identity(2))
    assert np.allclose(noisy_op.channel_matrix, real @ real)


def test_add_bad_type():
    ideal = cirq.Circuit([cirq.X.on(cirq.NamedQubit("Q"))])
    real = np.random.rand(4, 4)

    noisy_op = NoisyOperation(ideal, real)

    with pytest.raises(ValueError, match="must be a NoisyOperation"):
        noisy_op + ideal


def test_add_noisy_operation_no_channel_matrix():
    noisy_op1 = NoisyOperation(cirq.Circuit([cirq.X.on(cirq.NamedQubit("Q"))]))
    noisy_op2 = NoisyOperation(
        cirq.Circuit([cirq.X.on(cirq.NamedQubit("Q"))]),
        channel_matrix=np.random.rand(4, 4),
    )

    with pytest.raises(ValueError):
        (noisy_op1 + noisy_op2).channel_matrix


@pytest.mark.parametrize(
    "qreg",
    (
        cirq.LineQubit.range(5),
        cirq.GridQubit.square(5),
        [cirq.NamedQubit(str(i)) for i in range(5)],
    ),
)
@pytest.mark.parametrize("real", [np.zeros(shape=(4, 4)), None])
def test_on_each_single_qubit(qreg, real):
    noisy_ops = NoisyOperation.on_each(
        cirq.X, qubits=qreg, channel_matrix=real
    )

    assert len(noisy_ops) == len(qreg)

    for i, op in enumerate(noisy_ops):
        if real is not None:
            assert np.allclose(op.channel_matrix, real)
        assert op.num_qubits == 1
        assert list(op.circuit().all_qubits())[0] == qreg[i]


@pytest.mark.parametrize(
    "qubits",
    (
        [cirq.LineQubit.range(2), cirq.LineQubit.range(2, 4)],
        [cirq.LineQubit.range(5, 7), cirq.LineQubit.range(10, 12)],
        [cirq.GridQubit.rect(1, 2), cirq.GridQubit.rect(2, 1)],
    ),
)
@pytest.mark.parametrize("real", [np.zeros(shape=(16, 16)), None])
def test_on_each_multiple_qubits(qubits, real):
    noisy_ops = NoisyOperation.on_each(
        cirq.CNOT, qubits=qubits, channel_matrix=real
    )
    assert len(noisy_ops) == 2

    for i, op in enumerate(noisy_ops):
        if real is not None:
            assert np.allclose(op.channel_matrix, real)
        assert op.num_qubits == 2
        assert set(op.qubits) == set(qubits[i])


def test_on_each_multiple_qubits_bad_qubits_shape():
    real_cnot = np.zeros(shape=(16, 16))
    qubits = [cirq.LineQubit.range(3)]
    with pytest.raises(
        ValueError, match="Number of qubits in each register should be"
    ):
        NoisyOperation.on_each(
            cirq.CNOT, qubits=qubits, channel_matrix=real_cnot
        )


def test_on_each_bad_types():
    ideal = cirq.Circuit(cirq.I(cirq.LineQubit(0)))
    real = np.identity(4)
    with pytest.raises(TypeError, match="must be iterable"):
        NoisyOperation.on_each(
            ideal, qubits=cirq.NamedQubit("new"), channel_matrix=real
        )


@pytest.mark.parametrize(
    "qubit", (cirq.NamedQubit("New qubit"), cirq.GridQubit(2, 3))
)
@pytest.mark.parametrize("real", [np.zeros(shape=(4, 4)), None])
def test_transform_qubits_single_qubit(qubit, real):
    gate = cirq.H
    noisy_op = NoisyOperation.from_cirq(gate, real)

    assert set(noisy_op.qubits) != {qubit}
    noisy_op.transform_qubits(qubit)
    assert set(noisy_op.qubits) == {qubit}


@pytest.mark.parametrize(
    "qubits", (cirq.LineQubit.range(4, 6), cirq.GridQubit.rect(1, 2))
)
@pytest.mark.parametrize("real", [np.zeros(shape=(16, 16)), None])
def test_transform_qubits_multiple_qubits(qubits, real):
    qreg = [cirq.NamedQubit("Dummy 1"), cirq.NamedQubit("Dummy 2")]
    ideal = cirq.Circuit(cirq.ops.H.on(qreg[0]), cirq.ops.CNOT.on(*qreg))
    noisy_op = NoisyOperation(ideal, real)

    assert set(noisy_op.qubits) != set(qubits)
    if real is not None:
        assert np.allclose(noisy_op.channel_matrix, real)

    noisy_op.transform_qubits(qubits)
    assert set(noisy_op.qubits) == set(qubits)
    if real is not None:
        assert np.allclose(noisy_op.channel_matrix, real)


def test_transform_qubits_wrong_number():
    real = np.zeros(shape=(16, 16))
    qreg = [cirq.NamedQubit("Dummy 1"), cirq.NamedQubit("Dummy 2")]
    ideal = cirq.Circuit(cirq.ops.CNOT.on(*qreg))
    noisy_op = NoisyOperation(ideal, real)

    with pytest.raises(ValueError, match="Expected 2 qubits but received"):
        noisy_op.transform_qubits(qubits=[cirq.NamedQubit("new")])


def test_with_qubits():
    real = np.zeros(shape=(16, 16))
    qreg = [cirq.NamedQubit("Dummy 1"), cirq.NamedQubit("Dummy 2")]
    ideal = cirq.Circuit(cirq.ops.H.on(qreg[0]), cirq.ops.CNOT.on(*qreg))
    noisy_op = NoisyOperation(ideal, real)

    assert set(noisy_op.qubits) == set(qreg)
    assert np.allclose(noisy_op.channel_matrix, real)

    qubits = cirq.LineQubit.range(2)
    new_noisy_op = noisy_op.with_qubits(qubits)
    assert set(new_noisy_op.qubits) == set(qubits)
    assert np.allclose(new_noisy_op.channel_matrix, real)


@pytest.mark.parametrize("real", [np.zeros(shape=(4, 4)), None])
def test_extend_to_single_qubit(real):
    qbit, qreg = cirq.LineQubit(0), cirq.LineQubit.range(1, 10)
    ideal = cirq.Z.on(qbit)
    noisy_op_on_one_qubit = NoisyOperation.from_cirq(ideal, real)

    noisy_ops_on_all_qubits = noisy_op_on_one_qubit.extend_to(qreg)

    assert isinstance(noisy_ops_on_all_qubits, list)
    assert len(noisy_ops_on_all_qubits) == 10

    for op in noisy_ops_on_all_qubits:
        assert _equal(op.circuit(), cirq.Circuit(ideal))
        assert np.allclose(op.ideal_unitary, cirq.unitary(ideal))

        if real is not None:
            assert np.allclose(op.channel_matrix, real)


def test_noisy_operation_str():
    noisy_op = NoisyOperation.from_cirq(
        circuit=cirq.I, channel_matrix=np.identity(4)
    )
    assert isinstance(noisy_op.__str__(), str)


def test_noisy_basis_simple():
    rng = np.random.RandomState(seed=1)
    noisy_basis = NoisyBasis(
        NoisyOperation.from_cirq(
            circuit=cirq.I, channel_matrix=rng.rand(4, 4)
        ),
        NoisyOperation.from_cirq(
            circuit=cirq.X, channel_matrix=rng.rand(4, 4)
        ),
        NoisyOperation.from_cirq(
            circuit=cirq.Y, channel_matrix=rng.rand(4, 4)
        ),
        NoisyOperation.from_cirq(
            circuit=cirq.Z, channel_matrix=rng.rand(4, 4)
        ),
    )
    assert len(noisy_basis) == 4
    assert noisy_basis.all_qubits() == {cirq.LineQubit(0)}


def test_pyquil_noisy_basis():
    rng = np.random.RandomState(seed=1)

    noisy_basis = NoisyBasis(
        NoisyOperation(
            circuit=pyquil.Program(pyquil.gates.I(0)),
            channel_matrix=rng.rand(4, 4),
        ),
        NoisyOperation(
            circuit=pyquil.Program(pyquil.gates.Y(0)),
            channel_matrix=rng.rand(4, 4),
        ),
    )
    assert len(noisy_basis) == 2

    for op in noisy_basis.elements:
        assert isinstance(op.circuit(), pyquil.Program)
        assert isinstance(op._circuit, cirq.Circuit)


def test_qiskit_noisy_basis():
    rng = np.random.RandomState(seed=1)

    qreg = qiskit.QuantumRegister(1)
    xcirc = qiskit.QuantumCircuit(qreg)
    _ = xcirc.x(qreg)
    zcirc = qiskit.QuantumCircuit(qreg)
    _ = zcirc.z(qreg)

    noisy_basis = NoisyBasis(
        NoisyOperation(circuit=xcirc, channel_matrix=rng.rand(4, 4)),
        NoisyOperation(circuit=zcirc, channel_matrix=rng.rand(4, 4)),
    )
    assert len(noisy_basis) == 2

    for op in noisy_basis.elements:
        assert isinstance(op.circuit(), qiskit.QuantumCircuit)
        assert isinstance(op._circuit, cirq.Circuit)


@pytest.mark.parametrize(
    "element",
    (
        cirq.X,
        cirq.CNOT(*cirq.LineQubit.range(2)),
        pyquil.gates.H,
        pyquil.gates.CNOT(0, 1),
        qiskit.extensions.HGate,
        qiskit.extensions.CXGate,
    ),
)
def test_noisy_basis_bad_types(element):
    with pytest.raises(ValueError, match="must be of type `NoisyOperation`"):
        NoisyBasis(element)


def test_noisy_basis_add():
    rng = np.random.RandomState(seed=1)
    noisy_basis = NoisyBasis(
        NoisyOperation.from_cirq(
            circuit=cirq.I, channel_matrix=rng.rand(4, 4)
        ),
        NoisyOperation.from_cirq(
            circuit=cirq.X, channel_matrix=rng.rand(4, 4)
        ),
    )
    assert len(noisy_basis) == 2

    noisy_basis.add(
        NoisyOperation.from_cirq(
            circuit=cirq.Y, channel_matrix=rng.rand(4, 4)
        ),
        NoisyOperation.from_cirq(
            circuit=cirq.Z, channel_matrix=rng.rand(4, 4)
        ),
    )
    assert len(noisy_basis) == 4


def test_noisy_basis_add_bad_types():
    rng = np.random.RandomState(seed=1)
    noisy_basis = NoisyBasis(
        NoisyOperation.from_cirq(
            circuit=cirq.I, channel_matrix=rng.rand(4, 4)
        ),
        NoisyOperation.from_cirq(
            circuit=cirq.X, channel_matrix=rng.rand(4, 4)
        ),
    )

    with pytest.raises(TypeError, match="All basis elements must be of type"):
        noisy_basis.add(cirq.Y)


def test_extend_to_simple():
    rng = np.random.RandomState(seed=1)
    noisy_basis = NoisyBasis(
        NoisyOperation.from_cirq(
            circuit=cirq.I, channel_matrix=rng.rand(4, 4)
        ),
        NoisyOperation.from_cirq(
            circuit=cirq.X, channel_matrix=rng.rand(4, 4)
        ),
    )
    assert len(noisy_basis.elements) == 2

    noisy_basis.extend_to(cirq.LineQubit.range(1, 3))
    assert len(noisy_basis.elements) == 6


@pytest.mark.parametrize("length", (2, 3, 5))
def test_get_sequences_simple(length):
    rng = np.random.RandomState(seed=1)
    noisy_basis = NoisyBasis(
        NoisyOperation.from_cirq(
            circuit=cirq.I, channel_matrix=rng.rand(4, 4)
        ),
        NoisyOperation.from_cirq(
            circuit=cirq.X, channel_matrix=rng.rand(4, 4)
        ),
    )

    sequences = noisy_basis.get_sequences(length=length)
    assert all(isinstance(s, NoisyOperation) for s in sequences)
    assert len(sequences) == len(noisy_basis) ** length

    for sequence in sequences:
        assert len(sequence.circuit()) == length


def get_test_representation():
    ideal = cirq.Circuit(cirq.H(cirq.LineQubit(0)))

    noisy_xop = NoisyOperation.from_cirq(
        circuit=cirq.X, channel_matrix=np.zeros(shape=(4, 4))
    )
    noisy_zop = NoisyOperation.from_cirq(
        circuit=cirq.Z, channel_matrix=np.zeros(shape=(4, 4))
    )

    decomp = OperationRepresentation(
        ideal=ideal, basis_expansion={noisy_xop: 0.5, noisy_zop: -0.5}
    )
    return ideal, noisy_xop, noisy_zop, decomp


def test_representation_simple():
    ideal, noisy_xop, noisy_zop, decomp = get_test_representation()

    assert _equal(decomp.ideal, ideal)
    assert decomp.coeffs == (0.5, -0.5)
    assert np.allclose(decomp.distribution(), np.array([0.5, 0.5]))
    assert np.isclose(decomp.norm, 1.0)
    assert isinstance(decomp.basis_expansion, cirq.LinearDict)
    assert set(decomp.noisy_operations) == {noisy_xop, noisy_zop}


def test_representation_coeff_of():
    ideal, noisy_xop, noisy_zop, decomp = get_test_representation()

    assert np.isclose(decomp.coeff_of(noisy_xop), 0.5)
    assert np.isclose(decomp.coeff_of(noisy_zop), -0.5)


def test_representation_bad_type_for_basis_expansion():
    ideal = cirq.Circuit(cirq.H(cirq.LineQubit(0)))

    noisy_xop = NoisyOperation.from_cirq(
        circuit=cirq.X, channel_matrix=np.zeros(shape=(4, 4))
    )

    with pytest.raises(TypeError, match="All keys of `basis_expansion` must"):
        OperationRepresentation(
            ideal=ideal, basis_expansion=dict([(1.0, noisy_xop)])
        )


def test_representation_coeff_of_nonexistant_operation():
    qbit = cirq.LineQubit(0)
    ideal = cirq.Circuit(cirq.X(qbit))

    noisy_xop = NoisyOperation.from_cirq(
        circuit=cirq.X, channel_matrix=np.zeros(shape=(4, 4))
    )

    decomp = OperationRepresentation(
        ideal=ideal, basis_expansion=dict([(noisy_xop, 0.5)])
    )

    noisy_zop = NoisyOperation.from_cirq(
        circuit=cirq.Z, channel_matrix=np.zeros(shape=(4, 4))
    )
    with pytest.raises(ValueError, match="does not appear in the basis"):
        decomp.coeff_of(noisy_zop)


def test_representation_sign_of():
    _, noisy_xop, noisy_zop, decomp = get_test_representation()

    assert decomp.sign_of(noisy_xop) == 1.0
    assert decomp.sign_of(noisy_zop) == -1.0


def test_representation_sample():
    _, noisy_xop, noisy_zop, decomp = get_test_representation()

    for _ in range(10):
        noisy_op, sign, coeff = decomp.sample()
        assert sign in (-1, 1)
        assert coeff in (-0.5, 0.5)
        assert noisy_op in (noisy_xop, noisy_zop)

        assert decomp.sign_of(noisy_op) == sign
        assert decomp.coeff_of(noisy_op) == coeff


def test_representation_sample_seed():
    _, noisy_xop, noisy_zop, decomp = get_test_representation()

    seed1 = np.random.RandomState(seed=1)
    seed2 = np.random.RandomState(seed=1)
    for _ in range(10):
        _, sign1, coeff1 = decomp.sample(random_state=seed1)
        _, sign2, coeff2 = decomp.sample(random_state=seed2)

        assert sign1 == sign2
        assert np.isclose(coeff1, coeff2)


def test_representation_sample_bad_seed_type():
    _, _, _, decomp = get_test_representation()
    with pytest.raises(TypeError, match="should be of type"):
        decomp.sample(random_state=7)


def test_representation_sample_zero_coefficient():
    ideal = cirq.Circuit(cirq.H(cirq.LineQubit(0)))

    noisy_xop = NoisyOperation.from_cirq(
        circuit=cirq.X, channel_matrix=np.zeros(shape=(4, 4))
    )
    noisy_zop = NoisyOperation.from_cirq(
        circuit=cirq.Z, channel_matrix=np.zeros(shape=(4, 4))
    )

    decomp = OperationRepresentation(
        ideal=ideal,
        basis_expansion={
            noisy_xop: 0.5,
            noisy_zop: 0.0,  # This should never be sampled.
        },
    )

    random_state = np.random.RandomState(seed=1)
    for _ in range(500):
        noisy_op, sign, coeff = decomp.sample(random_state=random_state)
        assert sign == 1
        assert coeff == 0.5
        assert np.allclose(noisy_op.ideal_unitary, cirq.unitary(cirq.X))


def test_print_cirq_operation_representation():
    ideal = cirq.Circuit(cirq.H(cirq.LineQubit(0)))

    noisy_xop = NoisyOperation.from_cirq(
        circuit=cirq.X, channel_matrix=np.zeros(shape=(4, 4))
    )
    noisy_zop = NoisyOperation.from_cirq(
        circuit=cirq.Z, channel_matrix=np.zeros(shape=(4, 4))
    )
    # Positive first coefficient
    decomp = OperationRepresentation(
        ideal=ideal, basis_expansion={noisy_xop: 0.5, noisy_zop: 0.5,},
    )
    expected = r"0: ───H─── = 0.500*(0: ───X───)+0.500*(0: ───Z───)"
    assert str(decomp) == expected
    # Negative first coefficient
    decomp = OperationRepresentation(
        ideal=ideal, basis_expansion={noisy_xop: -0.5, noisy_zop: 1.5,},
    )
    expected = r"0: ───H─── = -0.500*(0: ───X───)+1.500*(0: ───Z───)"
    assert str(decomp) == expected
    # Empty representation
    decomp = OperationRepresentation(ideal=ideal, basis_expansion={})
    expected = r"0: ───H─── = 0.000"
    assert str(decomp) == expected
    # Small coefficient approximation
    decomp = OperationRepresentation(
        ideal=ideal, basis_expansion={noisy_xop: 1.00001, noisy_zop: 0.00001},
    )
    expected = r"0: ───H─── = 1.000*(0: ───X───)"
    assert str(decomp) == expected
    # Small coefficient approximation different position
    decomp = OperationRepresentation(
        ideal=ideal, basis_expansion={noisy_xop: 0.00001, noisy_zop: 1.00001},
    )
    expected = r"0: ───H─── = 1.000*(0: ───Z───)"
    # Small coefficient approximation different position
    decomp = OperationRepresentation(
        ideal=ideal, basis_expansion={noisy_xop: 0.00001},
    )
    expected = r"0: ───H─── = 0.000"
    assert str(decomp) == expected


def test_print_operation_representation_two_qubits():
    qreg = cirq.LineQubit.range(2)
    ideal = cirq.Circuit(cirq.CNOT(*qreg))

    noisy_a = NoisyOperation.from_cirq(
        circuit=cirq.Circuit(
            cirq.H.on_each(qreg), cirq.CNOT(*qreg), cirq.H.on_each(qreg)
        )
    )
    noisy_b = NoisyOperation.from_cirq(
        circuit=cirq.Circuit(
            cirq.Z.on_each(qreg), cirq.CNOT(*qreg), cirq.Z.on_each(qreg),
        )
    )
    decomp = OperationRepresentation(
        ideal=ideal, basis_expansion={noisy_a: 0.5, noisy_b: 0.5,},
    )
    print(str(decomp))
    expected = f"""
0: ───@───
      │
1: ───X─── ={" "}

0.500
0: ───H───@───H───
          │
1: ───H───X───H───

+0.500
0: ───Z───@───Z───
          │
1: ───Z───X───Z───"""
    # Remove initial newline
    expected = expected[1:]
    assert str(decomp) == expected


def test_print_operation_representation_two_qubits_neg():
    qreg = cirq.LineQubit.range(2)
    ideal = cirq.Circuit(cirq.CNOT(*qreg))

    noisy_a = NoisyOperation.from_cirq(
        circuit=cirq.Circuit(
            cirq.H.on_each(qreg[0]), cirq.CNOT(*qreg), cirq.H.on_each(qreg[1])
        )
    )
    noisy_b = NoisyOperation.from_cirq(
        circuit=cirq.Circuit(cirq.Z.on_each(qreg[1]))
    )
    decomp = OperationRepresentation(
        ideal=ideal, basis_expansion={noisy_a: -0.5, noisy_b: 1.5,},
    )
    print(str(decomp))
    expected = f"""
0: ───@───
      │
1: ───X─── ={" "}

-0.500
0: ───H───@───────
          │
1: ───────X───H───

+1.500
1: ───Z───"""
    # Remove initial newline
    expected = expected[1:]
    assert str(decomp) == expected


def test_equal_method_of_representations():
    q = cirq.LineQubit(0)
    ideal = cirq.Circuit(cirq.H(q))
    noisy_xop_a = NoisyOperation(
        circuit=cirq.Circuit(cirq.X(q)), channel_matrix=np.zeros(shape=(4, 4)),
    )
    noisy_zop_a = NoisyOperation(
        circuit=cirq.Circuit(cirq.Z(q)), channel_matrix=np.zeros(shape=(4, 4)),
    )
    rep_a = OperationRepresentation(
        ideal=ideal, basis_expansion={noisy_xop_a: 0.5, noisy_zop_a: 0.5},
    )
    noisy_xop_b = NoisyOperation(
        circuit=cirq.Circuit(cirq.X(q)), channel_matrix=np.ones(shape=(4, 4)),
    )
    noisy_zop_b = NoisyOperation(
        circuit=cirq.Circuit(cirq.Z(q)), channel_matrix=np.ones(shape=(4, 4)),
    )
    rep_b = OperationRepresentation(
        ideal=ideal, basis_expansion={noisy_xop_b: 0.5, noisy_zop_b: 0.5},
    )
    # Equal representation up to real superoperators
    assert rep_a == rep_b
    # Different ideal
    ideal_b = cirq.Circuit(cirq.X(q))
    rep_b = OperationRepresentation(
        ideal=ideal_b, basis_expansion={noisy_xop_b: 0.5, noisy_zop_b: 0.5},
    )
    assert rep_a != rep_b
    # Different type
    q_b = qiskit.QuantumRegister(1)
    ideal_b = qiskit.QuantumCircuit(q_b)
    ideal_b.x(q_b)
    rep_b = OperationRepresentation(
        ideal=ideal_b, basis_expansion={noisy_xop_b: 0.5, noisy_zop_b: 0.5},
    )
    assert rep_a != rep_b
    # Different length
    rep_b = OperationRepresentation(
        ideal=ideal, basis_expansion={noisy_xop_b: 0.5},
    )
    assert rep_a != rep_b
    # Different operations
    noisy_diff = NoisyOperation(circuit=cirq.Circuit(cirq.H(q)))
    rep_b = OperationRepresentation(
        ideal=ideal, basis_expansion={noisy_xop_b: 0.5, noisy_diff: 0.5},
    )
    assert rep_a != rep_b
    # Different coefficients
    rep_b = OperationRepresentation(
        ideal=ideal, basis_expansion={noisy_xop_b: 0.7, noisy_zop_b: 0.5},
    )
    assert rep_a != rep_b
