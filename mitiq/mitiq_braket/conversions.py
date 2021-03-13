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

from typing import List, Optional, Union,TYPE_CHECKING

import numpy as np

from cirq import Circuit, Gate as CirqGate, LineQubit, ops as cirq_ops, protocols
from cirq.linalg.decompositions import deconstruct_single_qubit_matrix_into_angles, kak_decomposition
from braket.circuits import gates as braket_gates, Circuit as BKCircuit, Instruction

if TYPE_CHECKING:
    import cirq


def from_braket(circuit: BKCircuit) -> "cirq.Circuit":
    """Returns a Cirq circuit equivalent to the input Braket circuit.

    Note: The returned Cirq circuit acts on cirq.LineQubit's with indices equal
    to the qubit indices of the Braket circuit.

    Args:
        circuit: Braket circuit to convert to a Cirq circuit.
    """
    return Circuit(
        _translate_braket_instruction_to_cirq_operation(instr)
        for instr in circuit.instructions
    )


def to_braket(circuit: "cirq.Circuit") -> BKCircuit:
    """Returns a Braket circuit equivalent to the input Cirq circuit.

    Args:
        circuit: Cirq circuit to convert to a Braket circuit.
    """
    braket_circuit = BKCircuit()

    for op in circuit.all_operations():
        for instr in _translate_cirq_operation_to_braket_instruction(op):
            braket_circuit.add_instruction(instr)
    return braket_circuit


def _translate_braket_instruction_to_cirq_operation(instr):
    """Converts the braket instruction to an equivalent Cirq operation.

    Args:
        instr: Braket instruction to convert.

    Raises:
        ValueError: If the instruction cannot be converted to Cirq.
    """
    qubits = [LineQubit(int(qubit)) for qubit in instr.target]
    gate = instr.operator

    # One-qubit non-parameterized gates.
    if isinstance(gate, braket_gates.I):
        yield cirq_ops.I.on(*qubits)
    elif isinstance(gate, braket_gates.X):
        yield cirq_ops.X.on(*qubits)
    elif isinstance(gate, braket_gates.Y):
        yield cirq_ops.Y.on(*qubits)
    elif isinstance(gate, braket_gates.Z):
        yield cirq_ops.Z.on(*qubits)
    elif isinstance(gate, braket_gates.H):
        yield cirq_ops.H.on(*qubits)
    elif isinstance(gate, braket_gates.S):
        yield cirq_ops.S.on(*qubits)
    elif isinstance(gate, braket_gates.Si):
        yield cirq_ops.S.on(*qubits) ** -1.0
    elif isinstance(gate, braket_gates.T):
        yield cirq_ops.T.on(*qubits)
    elif isinstance(gate, braket_gates.Ti):
        yield cirq_ops.T.on(*qubits) ** -1.0
    elif isinstance(gate, braket_gates.V):
        yield cirq_ops.X.on(*qubits) ** 0.5
    elif isinstance(gate, braket_gates.Vi):
        yield cirq_ops.X.on(*qubits) ** -0.5

    # One-qubit parameterized gates.
    elif isinstance(gate, braket_gates.Rx):
        yield cirq_ops.rx(gate.angle).on(*qubits)
    elif isinstance(gate, braket_gates.Ry):
        yield cirq_ops.ry(gate.angle).on(*qubits)
    elif isinstance(gate, braket_gates.Rz):
        yield cirq_ops.rz(gate.angle).on(*qubits)
    elif isinstance(gate, braket_gates.PhaseShift):
        yield cirq_ops.Z.on(*qubits) ** (gate.angle / np.pi)

    # Two-qubit non-parameterized gates.
    elif isinstance(gate, braket_gates.CNot):
        yield cirq_ops.CNOT.on(*qubits)
    elif isinstance(gate, braket_gates.Swap):
        yield cirq_ops.SWAP.on(*qubits)
    elif isinstance(gate, braket_gates.ISwap):
        yield cirq_ops.ISWAP.on(*qubits)
    elif isinstance(gate, braket_gates.CZ):
        yield cirq_ops.CZ.on(*qubits)
    elif isinstance(gate, braket_gates.CY):
        yield cirq_ops.S.on(qubits[1]) ** -1
        yield cirq_ops.CNOT.on(*qubits)
        yield cirq_ops.S.on(qubits[1])

    # Two-qubit parameterized gates.
    elif isinstance(gate, braket_gates.PSwap):
        raise ValueError  # TODO.
    elif isinstance(gate, braket_gates.XY):
        raise ValueError  # TODO
    elif isinstance(gate, braket_gates.CPhaseShift):
        yield cirq_ops.CZ.on(*qubits) ** (gate.angle / np.pi)
    elif isinstance(gate, braket_gates.CPhaseShift00):
        raise ValueError  # TODO.
    elif isinstance(gate, braket_gates.CPhaseShift01):
        raise ValueError  # TODO.
    elif isinstance(gate, braket_gates.CPhaseShift10):
        raise ValueError  # TODO.
    elif isinstance(gate, braket_gates.XX):
        raise ValueError  # TODO.
        # yield cirq_ops.XXPowGate(exponent=gate.angle / np.pi).on(*qubits)
    elif isinstance(gate, braket_gates.YY):
        raise ValueError  # TODO.
        # yield cirq_ops.YYPowGate(exponent=gate.angle / np.pi).on(*qubits)
    elif isinstance(gate, braket_gates.ZZ):
        raise ValueError  # TODO.
        # yield cirq_ops.ZZPowGate(exponent=gate.angle / np.pi).on(*qubits)

    # Three-qubit non-parameterized gates.
    elif isinstance(gate, braket_gates.CCNot):
        yield cirq_ops.TOFFOLI.on(*qubits)
    elif isinstance(gate, braket_gates.CSwap):
        yield cirq_ops.FREDKIN.on(*qubits)

    # Custom gates.
    elif isinstance(gate, braket_gates.Unitary):
        yield _map_custom_braket_gate_to_custom_cirq_gate(gate).on(*qubits)

    # Unknown gates.
    else:
        raise ValueError(
            f"Unable to convert the gate {gate} to Cirq. If you think this"
            " is a bug, you can open an issue on the Mitiq GitHub at"
            " https://github.com/unitaryfund/mitiq."
        )


def _translate_cirq_operation_to_braket_instruction(op: cirq_ops.Operation):
    """Converts the Cirq operation to an equivalent Braket instruction.

    Args:
        op: Cirq operation to convert.

    Raises:
        ValueError: If the operation cannot be converted to a Braket
            instruction.
    """
    nqubits = protocols.num_qubits(op)

    if nqubits == 1:
        return _translate_one_qubit_cirq_operation_to_braket_instruction(op)

    elif nqubits == 2:
        return _translate_two_qubit_cirq_operation_to_braket_instruction(op)

    elif nqubits == 3:
        qubits = [q.x for q in op.qubits]
        if isinstance(op.gate, cirq_ops.TOFFOLI):
            return [Instruction(braket_gates.CCNot(), target=qubits)]
        elif isinstance(op.gate, cirq_ops.FREDKIN):
            return [Instruction(braket_gates.CSwap(), target=qubits)]

    # Unsupported gates.
    else:
        raise ValueError(
            f"Unable to convert {op} to Braket. If you think this is a bug, "
            "you can open an issue on the Mitiq GitHub at"
            " https://github.com/unitaryfund/mitiq."
        )


def _translate_one_qubit_cirq_operation_to_braket_instruction(
    op: Union[np.ndarray, "cirq.Operation"], target: Optional[int] = None,
) -> List[Instruction]:
    """Translates a one-qubit Cirq operation to a (sequence of) Braket
    instruction(s) according to the following rules:

    1. Attempts to find a "standard translation" from Cirq to Braket.
        - e.g., checks if `op` is Pauli-X and, if so, returns the Braket X.

    2. If (1) is not successful, decomposes the unitary of `op` to
    Rz(theta) Ry(phi) Rz(lambda) and returns the series of rotations as Braket
    instructions.

    Args:
        op: One-qubit Cirq operation to translate.
        target: Optional
    """
    # Translate qubit index.
    if not isinstance(op, np.ndarray):
        target = op.qubits[0].x

    if target is None:
        raise ValueError(
            "Arg `target` must be specified when `op` is a matrix."
        )

    # Check common single-qubit unitaries.
    if isinstance(op.gate, cirq_ops.XPowGate):
        exponent = op.gate.exponent

        if np.isclose(exponent, 1.0) or np.isclose(exponent, -1.0):
            return [Instruction(braket_gates.X(), target=target)]
        elif np.isclose(exponent, 0.5):
            return [Instruction(braket_gates.V(), target=target)]
        elif np.isclose(exponent, -0.5):
            return [Instruction(braket_gates.Vi(), target=target)]

        return [Instruction(braket_gates.Rx(exponent * np.pi), target=target)]

    elif isinstance(op.gate, cirq_ops.YPowGate):
        exponent = op.gate.exponent

        if np.isclose(exponent, 1.0) or np.isclose(exponent, -1.0):
            return [Instruction(braket_gates.Y(), target=target)]

        return [Instruction(braket_gates.Ry(exponent * np.pi), target=target)]

    elif isinstance(op.gate, cirq_ops.ZPowGate):
        exponent = op.gate.exponent

        if np.isclose(exponent, 1.0) or np.isclose(exponent, -1.0):
            return [Instruction(braket_gates.Z(), target=target)]
        elif np.isclose(exponent, 0.5):
            return [Instruction(braket_gates.S(), target=target)]
        elif np.isclose(exponent, -0.5):
            return [Instruction(braket_gates.Si(), target=target)]
        elif np.isclose(exponent, 0.25):
            return [Instruction(braket_gates.T(), target=target)]
        elif np.isclose(exponent, -0.25):
            return [Instruction(braket_gates.Ti(), target=target)]

        return [Instruction(braket_gates.Rz(exponent * np.pi), target=target)]

    elif isinstance(op.gate, cirq_ops.HPowGate) and np.isclose(abs(op.gate.exponent), 1.0):
        return [Instruction(braket_gates.H(), target=target)]

    # Arbitrary single-qubit unitary decomposition.
    # TODO: This does not account for global phase.
    a, b, c = deconstruct_single_qubit_matrix_into_angles(protocols.unitary(op))
    return [
        Instruction(braket_gates.Rz(a), target=target),
        Instruction(braket_gates.Ry(b), target=target),
        Instruction(braket_gates.Rz(c), target=target),
    ]


def _translate_two_qubit_cirq_operation_to_braket_instruction(
    op: "cirq.Operation",
) -> List[Instruction]:
    """Translates a two-qubit Cirq operation to a (sequence of) Braket
    instruction(s) according to the following rules:

    1. Attempts to find a "standard translation" from Cirq to Braket.
        - e.g., checks if `op` is a CNOT and, if so, returns the Braket CNOT.

    2. If (1) is not successful, performs a KAK decomposition of the unitary of
    `op` to obtain a circuit

        ──A1──X^0.5───@───X^a───X──────────────────@───B1───
                      │         │                  │
        ──A2──────────X───Y^b───@───X^-0.5───Z^c───X───B2────

    where A1, A2, B1, and B2 are arbitrary single-qubit unitaries and a, b, c
    are floats.

    Args:
        op: Two-qubit Cirq operation to translate.
    """
    # Translate qubit indices.
    q1, q2 = [qubit.x for qubit in op.qubits]

    # TODO: Check common two-qubit unitaries.
    if isinstance(op.gate, cirq_ops.CNotPowGate) and np.isclose(abs(op.gate.exponent), 1.0):
        return [Instruction(braket_gates.CNot(), target=[q1, q2])]
    elif isinstance(op.gate, cirq_ops.CZPowGate) and np.isclose(abs(op.gate.exponent), 1.0):
        return [Instruction(braket_gates.CZ(), target=[q1, q2])]

    # Arbitrary two-qubit unitary decomposition.
    kak = kak_decomposition(protocols.unitary(op))
    A1, A2 = kak.single_qubit_operations_before

    x, y, z = kak.interaction_coefficients
    a = x * -2 / np.pi + 0.5
    b = y * -2 / np.pi + 0.5
    c = z * -2 / np.pi + 0.5

    B1, B2 = kak.single_qubit_operations_after

    return [
        *_translate_one_qubit_cirq_operation_to_braket_instruction(A1, target=q1),
        *_translate_one_qubit_cirq_operation_to_braket_instruction(A2, target=q2),
        Instruction(braket_gates.Rx(0.5 / np.pi), target=q1),
        Instruction(braket_gates.CNot(), target=[q1, q2]),
        Instruction(braket_gates.Rx(a / np.pi), target=q1),
        Instruction(braket_gates.Ry(b / np.pi), target=q2),
        Instruction(braket_gates.CNot(), target=[q2, q1]),
        Instruction(braket_gates.Rx(-0.5 / np.pi), target=q2),
        Instruction(braket_gates.Rz(c / np.pi), target=q2),
        Instruction(braket_gates.CNot(), target=[q1, q2]),
        *_translate_one_qubit_cirq_operation_to_braket_instruction(B1, target=q1),
        *_translate_one_qubit_cirq_operation_to_braket_instruction(B2, target=q2),
    ]


def _map_custom_braket_gate_to_custom_cirq_gate(gate) -> "cirq.Gate":
    """Returns a custom Cirq gate equivalent to the input Braket Unitary gate.

    Args:
        gate: Braket Unitary gate defined by a matrix.
    """
    unitary = gate.to_matrix()
    nqubits = int(np.log2(unitary.shape[0]))

    class CustomGate(CirqGate):
        def __init__(self):
            super(CustomGate, self)

        def _num_qubits_(self):
            return nqubits

        def _unitary_(self):
            return unitary

    return CustomGate()
