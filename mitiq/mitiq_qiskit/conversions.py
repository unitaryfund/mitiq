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

"""Functions to convert between Mitiq's internal circuit representation and
Qiskit's circuit representation.
"""

from typing import List, Optional, Tuple

import numpy as np

import cirq
from cirq.contrib.qasm_import import circuit_from_qasm
import qiskit
from qiskit.extensions import Barrier

from mitiq.utils import _simplify_circuit_exponents


QASMType = str


def _remove_barriers(circuit: qiskit.QuantumCircuit) -> qiskit.QuantumCircuit:
    """Returns a copy of the input circuit with all barriers removed.

    Args:
        circuit: Qiskit circuit to remove barriers from.
    """
    copy = circuit.copy()
    for instr in copy.data:
        gate = instr[0]
        if isinstance(gate, Barrier):
            copy.data.remove(instr)
    return copy


def _map_bit_index(
    bit_index: int, new_register_sizes: List[int]
) -> Tuple[int, int]:
    """Returns the register index and (qu)bit index in this register for the
    mapped bit_index.

    Args:
        bit_index: Index of (qu)bit in the original register.
        new_register_sizes: List of sizes of the new registers.

    Example:
        bit_index = 3, new_register_sizes = [2, 3]
        returns (1, 0), meaning the mapped (qu)bit is in the 1st new register
        and has index 0 in this register.
    """
    max_indices_in_registers = np.cumsum(new_register_sizes) - 1

    # Could be faster via bisection.
    register_index = None
    for i in range(len(max_indices_in_registers)):
        if bit_index <= max_indices_in_registers[i]:
            register_index = i
            break
    assert register_index is not None

    if register_index == 0:
        return register_index, bit_index

    return (
        register_index,
        bit_index - max_indices_in_registers[register_index - 1] - 1,
    )


def _map_bits(bits, new_register_sizes, new_registers):
    """Maps (qu)bits to new registers."""
    if len(new_registers) == 0:
        return bits

    indices = [bit.index for bit in bits]
    mapped_indices = [_map_bit_index(i, new_register_sizes) for i in indices]

    if isinstance(new_registers[0], qiskit.QuantumRegister):
        Bit = qiskit.circuit.Qubit
    else:
        Bit = qiskit.circuit.Clbit

    return [Bit(new_registers[i], j) for i, j in mapped_indices]


def _transform_registers(
    circuit: qiskit.QuantumCircuit,
    new_qregs: Optional[List[qiskit.QuantumRegister]] = None,
    new_cregs: Optional[List[qiskit.ClassicalRegister]] = None,
) -> None:
    """Transforms the quantum registers in the circuit to the new registers.

    Args:
        circuit: Qiskit circuit with one quantum register.
        new_qregs: The new quantum registers for the circuit.
        new_cregs: The new classical registers for the circuit.

    Raises:
        ValueError:
            * If the input circuit has more than one quantum register.
            * If the number of qubits in the new quantum registers does not
            match the number of qubits in the circuit.
            * If the number of bits in the new classical registers does not
            match the number of bits in the circuit.
    """
    if new_qregs is None and new_cregs is None:
        return

    if new_qregs is None:
        new_qregs = []

    if new_cregs is None:
        new_cregs = []

    if len(circuit.qregs) > 1:
        raise ValueError(
            "Input circuit is required to have <= 1 quantum register but has "
            f"{len(circuit.qregs)} quantum registers."
        )

    if len(circuit.cregs) > 1:
        raise ValueError(
            "Input circuit is required to have <= 1 classical register but has"
            f" {len(circuit.cregs)} classical registers."
        )

    qreg_sizes = [qreg.size for qreg in new_qregs]
    nqubits_in_circuit = sum(qreg.size for qreg in circuit.qregs)

    if len(qreg_sizes) and sum(qreg_sizes) != nqubits_in_circuit:
        raise ValueError(
            f"The circuit has {nqubits_in_circuit} qubits, but the provided "
            f"quantum registers have {sum(qreg_sizes)} qubits."
        )

    creg_sizes = [creg.size for creg in new_cregs]
    nbits_in_circuit = sum(creg.size for creg in circuit.cregs)

    if len(creg_sizes) and sum(creg_sizes) != nbits_in_circuit:
        raise ValueError(
            f"The circuit has {nbits_in_circuit} bits, but the provided "
            f"classical registers have {sum(creg_sizes)} bits."
        )

    # Assign the new registers.
    if len(qreg_sizes):
        circuit.qregs = list(new_qregs)
    if len(creg_sizes):
        circuit.cregs = list(new_cregs)

    # Map the (qu)bits in operations to the new (qu)bits.
    new_ops = []
    for op in circuit.data:
        gate, qubits, cbits = op

        new_qubits = _map_bits(qubits, qreg_sizes, new_qregs)
        new_cbits = _map_bits(cbits, creg_sizes, new_cregs)

        new_ops.append((gate, new_qubits, new_cbits))

    circuit.data = new_ops


def to_qasm(circuit: cirq.Circuit) -> QASMType:
    """Returns a QASM string representing the input Mitiq circuit.

    Args:
        circuit: Mitiq circuit to convert to a QASM string.

    Returns:
        QASMType: QASM string equivalent to the input Mitiq circuit.
    """
    # Simplify exponents of gates. For example, H**-1 is simplified to H.
    _simplify_circuit_exponents(circuit)
    return circuit.to_qasm()


def to_qiskit(
    circuit: cirq.Circuit,
    qregs: Optional[List[qiskit.QuantumRegister]] = None,
    cregs: Optional[List[qiskit.ClassicalRegister]] = None,
) -> qiskit.QuantumCircuit:
    """Returns a Qiskit circuit equivalent to the input Mitiq circuit.

    Args:
        circuit: Mitiq circuit to convert to a Qiskit circuit.
        qregs: Quantum registers of the returned Qiskit circuit. If none are
            provided, a single default register is used.
        cregs: Classical registers of the return Qiskit circuit. If none are
            provided, a single default register is used.

    Returns:
        Qiskit.QuantumCircuit object equivalent to the input Mitiq circuit.
    """
    # Base conversion.
    qiskit_circuit = qiskit.QuantumCircuit.from_qasm_str(to_qasm(circuit))

    # Assign register structure.
    _transform_registers(qiskit_circuit, new_qregs=qregs, new_cregs=cregs)

    return qiskit_circuit


def from_qiskit(circuit: qiskit.QuantumCircuit) -> cirq.Circuit:
    """Returns a Mitiq circuit equivalent to the input Qiskit circuit.

    Args:
        circuit: Qiskit circuit to convert to a Mitiq circuit.

    Returns:
        Mitiq circuit representation equivalent to the input Qiskit circuit.
    """
    return from_qasm(circuit.qasm())


def from_qasm(qasm: QASMType) -> cirq.Circuit:
    """Returns a Mitiq circuit equivalent to the input QASM string.

    Args:
        qasm: QASM string to convert to a Mitiq circuit.

    Returns:
        Mitiq circuit representation equivalent to the input QASM string.
    """
    qasm = _remove_barriers(qiskit.QuantumCircuit.from_qasm_str(qasm)).qasm()
    return circuit_from_qasm(qasm)
