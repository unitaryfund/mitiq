# Copyright (C) 2021 Unitary Fund
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

import pytest
import numpy as np

from braket.circuits import (
    Circuit as BKCircuit,
    gates as braket_gates,
    Instruction,
)
from cirq import Circuit, LineQubit, ops, unitary

from mitiq.mitiq_braket.conversions import from_braket
from mitiq.utils import _equal


def test_from_braket_bell_circuit():
    braket_circuit = BKCircuit().h(0).cnot(0, 1)
    cirq_circuit = from_braket(braket_circuit)

    expected_cirq_circuit = Circuit(
        ops.H(LineQubit(0)), ops.CNOT(*LineQubit.range(2))
    )
    assert _equal(cirq_circuit, expected_cirq_circuit)


def test_from_braket_non_parameterized_single_qubit_gates():
    braket_circuit = BKCircuit()
    instructions = [
        Instruction(braket_gates.I(), target=0),
        Instruction(braket_gates.X(), target=1),
        Instruction(braket_gates.Y(), target=2),
        Instruction(braket_gates.Z(), target=3),
        Instruction(braket_gates.H(), target=0),
        Instruction(braket_gates.S(), target=1),
        Instruction(braket_gates.Si(), target=2),
        Instruction(braket_gates.T(), target=3),
        Instruction(braket_gates.Ti(), target=0),
        Instruction(braket_gates.V(), target=1),
        Instruction(braket_gates.Vi(), target=2),
    ]
    for instr in instructions:
        braket_circuit.add_instruction(instr)
    cirq_circuit = from_braket(braket_circuit)

    for i, op in enumerate(cirq_circuit.all_operations()):
        assert np.allclose(instructions[i].operator.to_matrix(), unitary(op))

    qreg = LineQubit.range(4)
    expected_cirq_circuit = Circuit(
        ops.I(qreg[0]),
        ops.X(qreg[1]),
        ops.Y(qreg[2]),
        ops.Z(qreg[3]),
        ops.H(qreg[0]),
        ops.S(qreg[1]),
        ops.S(qreg[2]) ** -1,
        ops.T(qreg[3]),
        ops.T(qreg[0]) ** -1,
        ops.X(qreg[1]) ** 0.5,
        ops.X(qreg[2]) ** -0.5,
    )
    assert _equal(cirq_circuit, expected_cirq_circuit)


@pytest.mark.parametrize("qubit_index", (0, 3))
def test_from_braket_parameterized_single_qubit_gates(qubit_index):
    braket_circuit = BKCircuit()
    pgates = [
        braket_gates.Rx,
        braket_gates.Ry,
        braket_gates.Rz,
        braket_gates.PhaseShift,
    ]
    angles = np.random.RandomState(11).random(len(pgates))
    instructions = [
        Instruction(rot(a), target=qubit_index)
        for rot, a in zip(pgates, angles)
    ]
    for instr in instructions:
        braket_circuit.add_instruction(instr)
    cirq_circuit = from_braket(braket_circuit)

    for i, op in enumerate(cirq_circuit.all_operations()):
        assert np.allclose(instructions[i].operator.to_matrix(), unitary(op))

    qubit = LineQubit(qubit_index)
    expected_cirq_circuit = Circuit(
        ops.rx(angles[0]).on(qubit),
        ops.ry(angles[1]).on(qubit),
        ops.rz(angles[2]).on(qubit),
        ops.Z.on(qubit) ** (angles[3] / np.pi),
    )
    assert _equal(
        cirq_circuit, expected_cirq_circuit, require_qubit_equality=True
    )


def test_from_braket_non_parameterized_two_qubit_gates():
    braket_circuit = BKCircuit()
    instructions = [
        Instruction(braket_gates.CNot(), target=[2, 3]),
        Instruction(braket_gates.Swap(), target=[3, 4]),
        Instruction(braket_gates.ISwap(), target=[2, 3]),
        Instruction(braket_gates.CZ(), target=(3, 4)),
        Instruction(braket_gates.CY(), target=(2, 3)),
    ]
    for instr in instructions:
        braket_circuit.add_instruction(instr)
    cirq_circuit = from_braket(braket_circuit)

    qreg = LineQubit.range(2, 5)
    expected_cirq_circuit = Circuit(
        ops.CNOT(*qreg[:2]),
        ops.SWAP(*qreg[1:]),
        ops.ISWAP(*qreg[:2]),
        ops.CZ(*qreg[1:]),
        ops.ControlledGate(ops.Y).on(*qreg[:2]),
    )
    assert np.allclose(unitary(cirq_circuit), unitary(expected_cirq_circuit))


def test_from_braket_three_qubit_gates():
    braket_circuit = BKCircuit()
    instructions = [
        Instruction(braket_gates.CCNot(), target=[1, 2, 3]),
        Instruction(braket_gates.CSwap(), target=[1, 2, 3]),
    ]
    for instr in instructions:
        braket_circuit.add_instruction(instr)
    cirq_circuit = from_braket(braket_circuit)

    qreg = LineQubit.range(1, 4)
    expected_cirq_circuit = Circuit(ops.TOFFOLI(*qreg), ops.FREDKIN(*qreg))
    assert np.allclose(unitary(cirq_circuit), unitary(expected_cirq_circuit))
