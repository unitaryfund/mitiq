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
from cirq import Circuit
import pyquil
import qiskit

from mitiq.utils import _equal
from mitiq.pec.types import (
    NoisyBasis,
    NoisyOperation,
    OperationRepresentation,
)

icirq = Circuit(cirq.I(cirq.LineQubit(0)))
xcirq = Circuit(cirq.X(cirq.LineQubit(0)))
ycirq = Circuit(cirq.Y(cirq.LineQubit(0)))
zcirq = Circuit(cirq.Z(cirq.LineQubit(0)))
hcirq = Circuit(cirq.H(cirq.LineQubit(0)))
cnotcirq = Circuit(cirq.CNOT(cirq.LineQubit(0), cirq.LineQubit(1)))


def test_init_with_cirq_circuit():
    real = np.zeros(shape=(4, 4))
    noisy_op = NoisyOperation(zcirq, real)
    assert isinstance(noisy_op._circuit, cirq.Circuit)

    assert noisy_op.qubits == (cirq.LineQubit(0),)
    assert np.allclose(noisy_op.ideal_unitary, cirq.unitary(cirq.Z))
    assert np.allclose(noisy_op.channel_matrix, real)
    assert noisy_op.channel_matrix is not real

    assert noisy_op._native_type == "cirq"
    assert _equal(noisy_op._native_circuit, noisy_op.circuit)


@pytest.mark.parametrize(
    "qubit",
    (cirq.LineQubit(0), cirq.GridQubit(1, 2), cirq.NamedQubit("Qubit")),
)
def test_init_with_different_qubits(qubit):
    ideal_op = Circuit(cirq.H.on(qubit))
    real = np.zeros(shape=(4, 4))
    noisy_op = NoisyOperation(ideal_op, real)

    assert isinstance(noisy_op._circuit, cirq.Circuit)
    assert _equal(
        noisy_op.circuit,
        cirq.Circuit(ideal_op),
        require_qubit_equality=True,
    )
    assert noisy_op.qubits == (qubit,)
    assert np.allclose(noisy_op.ideal_unitary, cirq.unitary(ideal_op))
    assert np.allclose(noisy_op.channel_matrix, real)
    assert noisy_op.channel_matrix is not real

    assert noisy_op._native_type == "cirq"
    assert _equal(noisy_op._native_circuit, noisy_op.circuit)


def test_init_with_cirq_input():
    qreg = cirq.LineQubit.range(2)
    circ = cirq.Circuit(cirq.H.on(qreg[0]), cirq.CNOT.on(*qreg))
    real = np.zeros(shape=(16, 16))
    noisy_op = NoisyOperation(circ, real)

    assert isinstance(noisy_op._circuit, cirq.Circuit)
    assert _equal(noisy_op.circuit, circ, require_qubit_equality=True)
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
    assert _equal(noisy_op.circuit, cirq_circ)

    assert noisy_op.native_circuit == circ
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
    with pytest.raises(TypeError, match="Failed to convert to an internal"):
        NoisyOperation(circuit=gate, channel_matrix=rng.rand(4, 4))


def test_init_with_pyquil_program():
    circ = pyquil.Program(pyquil.gates.H(0), pyquil.gates.CNOT(0, 1))

    cirq_qreg = cirq.LineQubit.range(2)
    cirq_circ = cirq.Circuit(cirq.H.on(cirq_qreg[0]), cirq.CNOT.on(*cirq_qreg))

    real = np.zeros(shape=(16, 16))
    noisy_op = NoisyOperation(circ, real)
    assert isinstance(noisy_op._circuit, cirq.Circuit)
    assert _equal(noisy_op._circuit, cirq_circ)
    assert _equal(noisy_op.circuit, cirq_circ)

    assert noisy_op.native_circuit == circ
    assert noisy_op._native_circuit == circ
    assert noisy_op._native_type == "pyquil"

    assert np.allclose(noisy_op.ideal_unitary, cirq.unitary(cirq_circ))
    assert np.allclose(noisy_op.channel_matrix, real)
    assert noisy_op.channel_matrix is not real


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
    assert _equal(noisy_op.circuit, cirq_circ)

    assert noisy_op.native_circuit == circ
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


def test_noisy_operation_str():
    noisy_op = NoisyOperation(circuit=icirq, channel_matrix=np.identity(4))
    assert isinstance(noisy_op.__str__(), str)


def test_noisy_basis_simple():
    rng = np.random.RandomState(seed=1)
    noisy_basis = NoisyBasis(
        NoisyOperation(circuit=icirq, channel_matrix=rng.rand(4, 4)),
        NoisyOperation(circuit=xcirq, channel_matrix=rng.rand(4, 4)),
        NoisyOperation(circuit=ycirq, channel_matrix=rng.rand(4, 4)),
        NoisyOperation(circuit=zcirq, channel_matrix=rng.rand(4, 4)),
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
        assert isinstance(op.native_circuit, pyquil.Program)
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
        assert isinstance(op.native_circuit, qiskit.QuantumCircuit)
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
        NoisyOperation(circuit=icirq, channel_matrix=rng.rand(4, 4)),
        NoisyOperation(circuit=xcirq, channel_matrix=rng.rand(4, 4)),
    )
    assert len(noisy_basis) == 2

    noisy_basis.add(
        NoisyOperation(circuit=ycirq, channel_matrix=rng.rand(4, 4)),
        NoisyOperation(circuit=zcirq, channel_matrix=rng.rand(4, 4)),
    )
    assert len(noisy_basis) == 4


def test_noisy_basis_add_bad_types():
    rng = np.random.RandomState(seed=1)
    noisy_basis = NoisyBasis(
        NoisyOperation(icirq, channel_matrix=rng.rand(4, 4)),
        NoisyOperation(xcirq, channel_matrix=rng.rand(4, 4)),
    )

    with pytest.raises(TypeError, match="All basis elements must be of type"):
        noisy_basis.add(cirq.Y)


@pytest.mark.parametrize("length", (2, 3, 5))
def test_get_sequences_simple(length):
    rng = np.random.RandomState(seed=1)
    noisy_basis = NoisyBasis(
        NoisyOperation(circuit=icirq, channel_matrix=rng.rand(4, 4)),
        NoisyOperation(circuit=xcirq, channel_matrix=rng.rand(4, 4)),
    )

    sequences = noisy_basis.get_sequences(length=length)
    assert all(isinstance(s, NoisyOperation) for s in sequences)
    assert len(sequences) == len(noisy_basis) ** length

    for sequence in sequences:
        assert len(sequence.circuit) == length


def get_test_representation():
    ideal = cirq.Circuit(cirq.H(cirq.LineQubit(0)))

    noisy_xop = NoisyOperation(
        circuit=xcirq, channel_matrix=np.zeros(shape=(4, 4))
    )
    noisy_zop = NoisyOperation(
        circuit=zcirq, channel_matrix=np.zeros(shape=(4, 4))
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

    noisy_xop = NoisyOperation(
        circuit=xcirq, channel_matrix=np.zeros(shape=(4, 4))
    )

    with pytest.raises(TypeError, match="All keys of `basis_expansion` must"):
        OperationRepresentation(
            ideal=ideal, basis_expansion=dict([(1.0, noisy_xop)])
        )


def test_representation_coeff_of_nonexistant_operation():
    qbit = cirq.LineQubit(0)
    ideal = cirq.Circuit(cirq.X(qbit))

    noisy_xop = NoisyOperation(
        circuit=xcirq, channel_matrix=np.zeros(shape=(4, 4))
    )

    decomp = OperationRepresentation(
        ideal=ideal, basis_expansion=dict([(noisy_xop, 0.5)])
    )

    noisy_zop = NoisyOperation(
        circuit=zcirq, channel_matrix=np.zeros(shape=(4, 4))
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

    noisy_xop = NoisyOperation(
        circuit=xcirq, channel_matrix=np.zeros(shape=(4, 4))
    )
    noisy_zop = NoisyOperation(
        circuit=zcirq, channel_matrix=np.zeros(shape=(4, 4))
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

    noisy_xop = NoisyOperation(
        circuit=xcirq, channel_matrix=np.zeros(shape=(4, 4))
    )
    noisy_zop = NoisyOperation(
        circuit=zcirq, channel_matrix=np.zeros(shape=(4, 4))
    )
    # Positive first coefficient
    decomp = OperationRepresentation(
        ideal=ideal,
        basis_expansion={
            noisy_xop: 0.5,
            noisy_zop: 0.5,
        },
    )
    expected = r"0: ───H─── = 0.500*(0: ───X───)+0.500*(0: ───Z───)"
    assert str(decomp) == expected
    # Negative first coefficient
    decomp = OperationRepresentation(
        ideal=ideal,
        basis_expansion={
            noisy_xop: -0.5,
            noisy_zop: 1.5,
        },
    )
    expected = r"0: ───H─── = -0.500*(0: ───X───)+1.500*(0: ───Z───)"
    assert str(decomp) == expected
    # Empty representation
    decomp = OperationRepresentation(ideal=ideal, basis_expansion={})
    expected = r"0: ───H─── = 0.000"
    assert str(decomp) == expected
    # Small coefficient approximation
    decomp = OperationRepresentation(
        ideal=ideal,
        basis_expansion={noisy_xop: 1.00001, noisy_zop: 0.00001},
    )
    expected = r"0: ───H─── = 1.000*(0: ───X───)"
    assert str(decomp) == expected
    # Small coefficient approximation different position
    decomp = OperationRepresentation(
        ideal=ideal,
        basis_expansion={noisy_xop: 0.00001, noisy_zop: 1.00001},
    )
    expected = r"0: ───H─── = 1.000*(0: ───Z───)"
    # Small coefficient approximation different position
    decomp = OperationRepresentation(
        ideal=ideal,
        basis_expansion={noisy_xop: 0.00001},
    )
    expected = r"0: ───H─── = 0.000"
    assert str(decomp) == expected


def test_print_operation_representation_two_qubits():
    qreg = cirq.LineQubit.range(2)
    ideal = cirq.Circuit(cirq.CNOT(*qreg))

    noisy_a = NoisyOperation(
        circuit=cirq.Circuit(
            cirq.H.on_each(qreg), cirq.CNOT(*qreg), cirq.H.on_each(qreg)
        )
    )
    noisy_b = NoisyOperation(
        circuit=cirq.Circuit(
            cirq.Z.on_each(qreg),
            cirq.CNOT(*qreg),
            cirq.Z.on_each(qreg),
        )
    )
    decomp = OperationRepresentation(
        ideal=ideal,
        basis_expansion={
            noisy_a: 0.5,
            noisy_b: 0.5,
        },
    )
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

    noisy_a = NoisyOperation(
        circuit=cirq.Circuit(
            cirq.H.on_each(qreg[0]), cirq.CNOT(*qreg), cirq.H.on_each(qreg[1])
        )
    )
    noisy_b = NoisyOperation(circuit=cirq.Circuit(cirq.Z.on_each(qreg[1])))
    decomp = OperationRepresentation(
        ideal=ideal,
        basis_expansion={
            noisy_a: -0.5,
            noisy_b: 1.5,
        },
    )
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
        circuit=cirq.Circuit(cirq.X(q)),
        channel_matrix=np.zeros(shape=(4, 4)),
    )
    noisy_zop_a = NoisyOperation(
        circuit=cirq.Circuit(cirq.Z(q)),
        channel_matrix=np.zeros(shape=(4, 4)),
    )
    rep_a = OperationRepresentation(
        ideal=ideal,
        basis_expansion={noisy_xop_a: 0.5, noisy_zop_a: 0.5},
    )
    noisy_xop_b = NoisyOperation(
        circuit=cirq.Circuit(cirq.X(q)),
        channel_matrix=np.ones(shape=(4, 4)),
    )
    noisy_zop_b = NoisyOperation(
        circuit=cirq.Circuit(cirq.Z(q)),
        channel_matrix=np.ones(shape=(4, 4)),
    )
    rep_b = OperationRepresentation(
        ideal=ideal,
        basis_expansion={noisy_xop_b: 0.5, noisy_zop_b: 0.5},
    )
    # Equal representation up to real superoperators
    assert rep_a == rep_b
    # Different ideal
    ideal_b = cirq.Circuit(cirq.X(q))
    rep_b = OperationRepresentation(
        ideal=ideal_b,
        basis_expansion={noisy_xop_b: 0.5, noisy_zop_b: 0.5},
    )
    assert rep_a != rep_b
    # Different type
    q_b = qiskit.QuantumRegister(1)
    ideal_b = qiskit.QuantumCircuit(q_b)
    ideal_b.x(q_b)
    rep_b = OperationRepresentation(
        ideal=ideal_b,
        basis_expansion={noisy_xop_b: 0.5, noisy_zop_b: 0.5},
    )
    assert rep_a != rep_b
    # Different length
    rep_b = OperationRepresentation(
        ideal=ideal,
        basis_expansion={noisy_xop_b: 0.5},
    )
    assert rep_a != rep_b
    # Different operations
    noisy_diff = NoisyOperation(circuit=cirq.Circuit(cirq.H(q)))
    rep_b = OperationRepresentation(
        ideal=ideal,
        basis_expansion={noisy_xop_b: 0.5, noisy_diff: 0.5},
    )
    assert rep_a != rep_b
    # Different coefficients
    rep_b = OperationRepresentation(
        ideal=ideal,
        basis_expansion={noisy_xop_b: 0.7, noisy_zop_b: 0.5},
    )
    assert rep_a != rep_b
