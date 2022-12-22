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
import copy

import numpy as np
import pytest

import cirq
import qiskit

from mitiq.utils import _equal
from mitiq.interface.mitiq_qiskit.conversions import (
    to_qasm,
    to_qiskit,
    from_qasm,
    from_qiskit,
    _map_bit_index,
    _transform_registers,
    _measurement_order,
    _remove_qasm_barriers,
    _add_identity_to_idle,
    _remove_identity_from_idle,
)
from mitiq.interface import convert_to_mitiq


def _multi_reg_circuits():
    """Returns a circuit with multiple registers
    and an equivalent circuit with a single register."""
    qregs = [qiskit.QuantumRegister(1, f"q{j}") for j in range(4)]
    circuit_multi_reg = qiskit.QuantumCircuit(*qregs)
    for q in qregs[1:]:
        circuit_multi_reg.x(q)
        circuit_multi_reg.x(3)
    qreg = qiskit.QuantumRegister(4, "q")
    circuit_single_reg = qiskit.QuantumCircuit(qreg)
    for q in qreg[1:]:
        circuit_single_reg.x(q)
        circuit_single_reg.x(3)

    return circuit_multi_reg, circuit_single_reg


def test_remove_qasm_barriers():
    assert (
        _remove_qasm_barriers(
            """
// quantum ripple-carry adder from Cuccaro et al, quant-ph/0410184
OPENQASM 2.0;
include "qelib1.inc";
include "barrier.inc";
include ";barrier.inc";
gate majority a,b,c
{
  cx c,b;
  cx c,a;
  ccx a,b,c;
}
gate barrier1 a,a,a,a,a{
    barrier x,y,z;
    barrier1;
    barrier;
}
gate unmaj a,b,c
{
  ccx a,b,c;
  cx c,a;
  cx a,b;
}
qreg cin[1];
qreg a[4];
qreg b[4];
// barrier;
qreg cout[1]; barrier x,y,z;
creg ans[5];
// set input states
x a[0]; // a = 0001
x b;    // b = 1111
// add a to b, storing result in b
majority cin[0],b[0],a[0];
majority a[0],b[1],a[1];
majority a[1],b[2],a[2];
majority a[2],b[3],a[3];
cx a[3],cout[0];
unmaj a[2],b[3],a[3];
unmaj a[1],b[2],a[2];
unmaj a[0],b[1],a[1];
unmaj cin[0],b[0],a[0];
measure b[0] -> ans[0];
measure b[1] -> ans[1];
measure b[2] -> ans[2];
measure b[3] -> ans[3];
measure cout[0] -> ans[4];
// quantum ripple-carry adder from Cuccaro et al, quant-ph/0410184
"""
        )
        == """
// quantum ripple-carry adder from Cuccaro et al, quant-ph/0410184
OPENQASM 2.0;
include "qelib1.inc";
include "barrier.inc";
include ";barrier.inc";
gate majority a,b,c
{
  cx c,b;
  cx c,a;
  ccx a,b,c;
}
gate barrier1 a,a,a,a,a{
    barrier1;
}
gate unmaj a,b,c
{
  ccx a,b,c;
  cx c,a;
  cx a,b;
}
qreg cin[1];
qreg a[4];
qreg b[4];
// barrier;
qreg cout[1];
creg ans[5];
// set input states
x a[0]; // a = 0001
x b;    // b = 1111
// add a to b, storing result in b
majority cin[0],b[0],a[0];
majority a[0],b[1],a[1];
majority a[1],b[2],a[2];
majority a[2],b[3],a[3];
cx a[3],cout[0];
unmaj a[2],b[3],a[3];
unmaj a[1],b[2],a[2];
unmaj a[0],b[1],a[1];
unmaj cin[0],b[0],a[0];
measure b[0] -> ans[0];
measure b[1] -> ans[1];
measure b[2] -> ans[2];
measure b[3] -> ans[3];
measure cout[0] -> ans[4];
// quantum ripple-carry adder from Cuccaro et al, quant-ph/0410184
"""
    )


def test_bell_state_to_from_circuits():
    """Tests cirq.Circuit --> qiskit.QuantumCircuit --> cirq.Circuit
    with a Bell state circuit.
    """
    qreg = cirq.LineQubit.range(2)
    cirq_circuit = cirq.Circuit(
        [cirq.ops.H.on(qreg[0]), cirq.ops.CNOT.on(qreg[0], qreg[1])]
    )
    qiskit_circuit = to_qiskit(cirq_circuit)  # Qiskit from Cirq
    circuit_cirq = from_qiskit(qiskit_circuit)  # Cirq from Qiskit
    assert _equal(cirq_circuit, circuit_cirq)


def test_bell_state_to_from_qasm():
    """Tests cirq.Circuit --> QASM string --> cirq.Circuit
    with a Bell state circuit.
    """
    qreg = cirq.LineQubit.range(2)
    cirq_circuit = cirq.Circuit(
        [cirq.ops.H.on(qreg[0]), cirq.ops.CNOT.on(qreg[0], qreg[1])]
    )
    qasm = to_qasm(cirq_circuit)  # Qasm from Cirq
    circuit_cirq = from_qasm(qasm)
    assert _equal(cirq_circuit, circuit_cirq)


def test_random_circuit_to_from_circuits():
    """Tests cirq.Circuit --> qiskit.QuantumCircuit --> cirq.Circuit
    with a random two-qubit circuit.
    """
    cirq_circuit = cirq.testing.random_circuit(
        qubits=2, n_moments=10, op_density=0.99, random_state=1
    )
    qiskit_circuit = to_qiskit(cirq_circuit)
    circuit_cirq = from_qiskit(qiskit_circuit)
    assert cirq.equal_up_to_global_phase(
        cirq_circuit.unitary(), circuit_cirq.unitary()
    )


def test_random_circuit_to_from_qasm():
    """Tests cirq.Circuit --> QASM string --> cirq.Circuit
    with a random one-qubit circuit.
    """
    cirq_circuit = cirq.testing.random_circuit(
        qubits=2, n_moments=10, op_density=0.99, random_state=2
    )
    qasm = to_qasm(cirq_circuit)
    circuit_cirq = from_qasm(qasm)
    assert cirq.equal_up_to_global_phase(
        cirq_circuit.unitary(), circuit_cirq.unitary()
    )


@pytest.mark.parametrize("as_qasm", (True, False))
def test_convert_with_barrier(as_qasm):
    """Tests converting a Qiskit circuit with a barrier to a Cirq circuit."""
    n = 5
    qiskit_circuit = qiskit.QuantumCircuit(qiskit.QuantumRegister(n))
    qiskit_circuit.barrier()

    if as_qasm:
        cirq_circuit = from_qasm(qiskit_circuit.qasm())
    else:
        cirq_circuit = from_qiskit(qiskit_circuit)

    assert _equal(cirq_circuit, cirq.Circuit())


@pytest.mark.parametrize("as_qasm", (True, False))
def test_convert_with_multiple_barriers(as_qasm):
    """Tests converting a Qiskit circuit with barriers to a Cirq circuit."""
    n = 1
    num_ops = 10

    qreg = qiskit.QuantumRegister(n)
    qiskit_circuit = qiskit.QuantumCircuit(qreg)
    for _ in range(num_ops):
        qiskit_circuit.h(qreg)
        qiskit_circuit.barrier()

    if as_qasm:
        cirq_circuit = from_qasm(qiskit_circuit.qasm())
    else:
        cirq_circuit = from_qiskit(qiskit_circuit)

    qbit = cirq.LineQubit(0)
    correct = cirq.Circuit(cirq.ops.H.on(qbit) for _ in range(num_ops))
    assert _equal(cirq_circuit, correct)


@pytest.mark.parametrize("reg_sizes", [[2, 4, 1, 6], [5, 4, 2], [6]])
def test_map_bit_index(reg_sizes):
    expected_register_index = 0
    expected_mapped_index = 0
    for bit_index in range(sum(reg_sizes)):
        register_index, mapped_index = _map_bit_index(bit_index, reg_sizes)

        assert register_index == expected_register_index
        assert mapped_index == expected_mapped_index

        expected_mapped_index += 1
        if bit_index == sum(reg_sizes[: expected_register_index + 1]) - 1:
            expected_register_index += 1
            expected_mapped_index = 0


@pytest.mark.parametrize("nqubits", [1, 5])
@pytest.mark.parametrize("with_ops", [True, False])
@pytest.mark.parametrize("measure", [True, False])
def test_transform_qregs_one_qubit_ops(nqubits, with_ops, measure):
    qreg = qiskit.QuantumRegister(nqubits)
    circ = qiskit.QuantumCircuit(qreg)
    if with_ops:
        circ.h(qreg)
    if measure:
        circ.add_register(qiskit.ClassicalRegister(nqubits))
        circ.measure(qreg, circ.cregs[0])

    orig = circ.copy()
    assert circ.qregs == [qreg]

    new_qregs = [qiskit.QuantumRegister(1) for _ in range(nqubits)]
    _transform_registers(circ, new_qregs=new_qregs)

    assert circ.qregs == new_qregs
    assert circ.cregs == orig.cregs
    assert _equal(from_qiskit(circ), from_qiskit(orig))


@pytest.mark.parametrize("new_reg_sizes", [[1], [1, 2], [2, 1], [1, 1, 1]])
def test_transform_qregs_two_qubit_ops(new_reg_sizes):
    nqubits = sum(new_reg_sizes)
    circ = to_qiskit(
        cirq.testing.random_circuit(
            nqubits, n_moments=5, op_density=1, random_state=1
        )
    )
    orig = circ.copy()

    new_qregs = [qiskit.QuantumRegister(s) for s in new_reg_sizes]
    _transform_registers(circ, new_qregs=new_qregs)

    assert circ.qregs == new_qregs
    assert circ.cregs == orig.cregs
    assert _equal(from_qiskit(circ), from_qiskit(orig))


@pytest.mark.parametrize("new_reg_sizes", [[1], [1, 2], [2, 1], [1, 1, 1]])
@pytest.mark.parametrize("measure", [True, False])
def test_transform_qregs_random_circuit(new_reg_sizes, measure):
    nbits = sum(new_reg_sizes)
    circ = to_qiskit(
        cirq.testing.random_circuit(
            nbits, n_moments=5, op_density=1, random_state=10
        )
    )
    creg = qiskit.ClassicalRegister(nbits)
    circ.add_register(creg)
    if measure:
        circ.measure(circ.qregs[0], creg)
    orig = circ.copy()

    new_qregs = [qiskit.QuantumRegister(s) for s in new_reg_sizes]
    _transform_registers(circ, new_qregs=new_qregs)

    assert circ.qregs == new_qregs
    assert _equal(from_qiskit(circ), from_qiskit(orig))


def test_transform_qregs_no_new_qregs():
    qreg = qiskit.QuantumRegister(5)
    circ = qiskit.QuantumCircuit(qreg)
    _transform_registers(circ, new_qregs=None)
    assert circ.qregs == [qreg]


def test_transform_registers_too_few_qubits():
    circ = qiskit.QuantumCircuit(qiskit.QuantumRegister(2))
    new_qregs = [qiskit.QuantumRegister(1)]

    with pytest.raises(ValueError):
        _transform_registers(circ, new_qregs=new_qregs)


def test_transform_registers_adds_idle_qubits():
    """Tests transforming registers in a circuit with n qubits to a circuit
    with m > n qubits.
    """
    qreg = qiskit.QuantumRegister(1)
    creg = qiskit.ClassicalRegister(1)
    circuit = qiskit.QuantumCircuit(qreg, creg)
    circuit.x(qreg[0])
    circuit.measure(qreg[0], creg[0])

    assert len(circuit.qregs) == 1
    assert circuit.num_qubits == 1
    old_data = copy.deepcopy(circuit.data)

    _transform_registers(circuit, new_qregs=[qreg, qiskit.QuantumRegister(4)])

    assert len(circuit.qregs) == 2
    assert circuit.num_qubits == 5
    assert circuit.data == old_data


def test_transform_registers_wrong_reg_number():
    nqubits = 2
    circ = qiskit.QuantumCircuit(qiskit.QuantumRegister(nqubits))
    new_qregs = [qiskit.QuantumRegister(1) for _ in range(2 * nqubits)]
    circ.add_register(*new_qregs)

    with pytest.raises(ValueError):
        _transform_registers(circ, new_qregs=new_qregs)


@pytest.mark.parametrize("size", [5])
def test_measurement_order(size):
    q, c = qiskit.QuantumRegister(size), qiskit.ClassicalRegister(size)
    circuit = qiskit.QuantumCircuit(q, c)

    index_order = [int(i) for i in np.random.RandomState(1).permutation(size)]
    for i in index_order:
        circuit.measure(q[i], c[i])

    order = _measurement_order(circuit)
    assert order == [(q[i], c[i]) for i in index_order]


def test_add_identity_to_idle():
    circuit = qiskit.QuantumCircuit(9)
    circuit.x([0, 8])
    circuit.cx(0, 8)
    expected_idle_qubits = circuit.qubits[1:-1]

    idle_qubits = _add_identity_to_idle(circuit)
    id_qubits = []
    for gates, qubits, cargs in circuit.get_instructions("id"):
        for qubit in qubits:
            id_qubits.append(qubit)
    assert idle_qubits == set(expected_idle_qubits)
    assert set(id_qubits) == idle_qubits


def test_remove_identity_from_idle():
    idle_indices = set(range(1, 8))
    circuit = qiskit.QuantumCircuit(9)
    circuit.x([0, 8])
    circuit.cx(0, 8)
    _remove_identity_from_idle(circuit, idle_indices)
    id_indices = []
    for gates, qubits, cargs in circuit.get_instructions("id"):
        for qubit in qubits:
            id_indices.append(qubit.index)
    assert id_indices == []


def test_add_identity_to_idle_with_multiple_registers():
    """Tests idle qubits are correctly detected even with many registers."""
    circuit_multi_reg, circuit_single_reg = _multi_reg_circuits()
    _add_identity_to_idle(circuit_multi_reg)
    _add_identity_to_idle(circuit_single_reg)

    # The result should be the same for both types of registers
    assert _equal(
        convert_to_mitiq(circuit_multi_reg)[0],
        convert_to_mitiq(circuit_single_reg)[0],
        require_qubit_equality=False,  # Qubit names can be different
    )


def test_remove_identity_from_idle_with_multiple_registers():
    """Tests identities are correctly removed even with many registers."""
    circuit_multi_reg, circuit_single_reg = _multi_reg_circuits()

    idle_qubits_multi = _add_identity_to_idle(circuit_multi_reg.copy())
    idle_qubits_single = _add_identity_to_idle(circuit_single_reg.copy())

    _remove_identity_from_idle(circuit_multi_reg, idle_qubits_multi)
    _remove_identity_from_idle(circuit_single_reg, idle_qubits_single)

    # Adding and removing identities should preserve the input
    input_multi, input_single = _multi_reg_circuits()
    assert circuit_multi_reg == input_multi
    assert circuit_single_reg == input_single
