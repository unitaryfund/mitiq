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

from typing import TYPE_CHECKING

import numpy as np

from cirq import Circuit, Gate as CirqGate, LineQubit, ops as cirq_ops
from braket.circuits import gates as braket_gates, Circuit as BKCircuit

if TYPE_CHECKING:
    import cirq


def _map_custom_braket_gate_to_custom_cirq_gate(gate) -> CirqGate:
    """Returns a custom Cirq gate equivalent to the input Braket Unitary gate.

    Args:
        gate: Braket Unitary gate defined by a matrix.
    """
    unitary = gate.to_matrix()
    nqubits = int(np.log2(unitary.shape[0]))

    class CustomGate(cirq.Gate):
        def __init__(self):
            super(CustomGate, self)

        def _num_qubits_(self):
            return nqubits

        def _unitary_(self):
            return unitary

    return CustomGate()


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
        yield cirq_ops.S.on(*qubits) ** -1
    elif isinstance(gate, braket_gates.T):
        yield cirq_ops.T.on(*qubits)
    elif isinstance(gate, braket_gates.Ti):
        yield cirq_ops.T.on(*qubits) ** -1
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
        pass  # TODO.
    elif isinstance(gate, braket_gates.CPhaseShift):
        yield cirq_ops.CZ.on(*qubits) ** (gate.angle / np.pi)
    elif isinstance(gate, braket_gates.CPhaseShift00):
        raise ValueError  # TODO.
    elif isinstance(gate, braket_gates.CPhaseShift01):
        yield cirq_ops.CZ.on(*qubits[::-1]) ** (gate.angle / np.pi)
    elif isinstance(gate, braket_gates.CPhaseShift10):
        raise ValueError  # TODO.
    elif isinstance(gate, braket_gates.XX):
        raise ValueError  # TODO
    elif isinstance(gate, braket_gates.YY):
        raise ValueError  # TODO
    elif isinstance(gate, braket_gates.ZZ):
        raise ValueError  # TODO

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
            " is a bug, you can open an issue on the Mitiq Github at"
            " https://github.com/unitaryfund/mitiq."
        )


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
    pass
