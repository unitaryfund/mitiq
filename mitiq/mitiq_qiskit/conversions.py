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

from typing import List, Optional, Tuple, Union

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

    Note:
        The bit_index is assumed to come from a circuit with 1 or n registers
        where n is the maximum bit_index.
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


def _map_bits(
    bits: List[Union[qiskit.circuit.Qubit, qiskit.circuit.Clbit]],
    registers: List[Union[qiskit.QuantumRegister, qiskit.ClassicalRegister]],
    new_register_sizes: List[int],
    new_registers: List[
        Union[qiskit.QuantumRegister, qiskit.ClassicalRegister]
    ],
) -> List[Union[qiskit.circuit.Qubit, qiskit.circuit.Clbit]]:
    """Maps (qu)bits to new registers. Assumes the input ``bits`` come from
    a single register or n registers, where n is the number of bits.

    Args:
        bits: A list of (qu)bits to map.
        registers: The registers that the ``bits`` come from.
        new_register_sizes: The size(s) of the new registers to map to.
            Note: These can be determined from ``new_registers``, but this
            helper function is only called from ``_map_bits`` where the sizes
            are already computed.
        new_registers: The new registers to map the ``bits`` to .

    Returns:
        The input ``bits`` mapped to the ``new_registers``.
    """
    if len(new_registers) == 0:
        return bits

    # Only two support cases:
    if len(registers) == 1:
        # Case where there are n bits in a single register.
        indices = [bit.index for bit in bits]
    else:
        # Case where there are n single-bit registers.
        indices = [registers.index(bit.register) for bit in bits]

    mapped_indices = [_map_bit_index(i, new_register_sizes) for i in indices]

    if isinstance(new_registers[0], qiskit.QuantumRegister):
        Bit = qiskit.circuit.Qubit
    else:
        Bit = qiskit.circuit.Clbit

    return [Bit(new_registers[i], j) for i, j in mapped_indices]


def _measurement_order(circuit: qiskit.QuantumCircuit):
    """Returns the left-to-right measurement order in the circuit.

    The "measurement order" is a list of tuples (qubit, bit) involved in
    measurements ordered as they appear going left-to-right through the circuit
    (i.e., iterating through circuit.data). The purpose of this is to be able
    to do

    >>> for (qubit, bit) in _measurement_order(circuit):
    >>>     other_circuit.measure(qubit, bit)

    which ensures ``other_circuit`` has the same measurement order as
    ``circuit``, assuming ``other_circuit`` has the same register(s) as
    ``circuit``.

    Args:
        circuit: Qiskit circuit to get the measurement order of.
    """
    order = []
    for (gate, qubits, cbits) in circuit.data:
        if isinstance(gate, qiskit.circuit.Measure):
            if len(qubits) != 1 or len(cbits) != 1:
                raise ValueError(
                    f"Only measurements with one qubit and one bit are "
                    f"supported, but this measurement has {len(qubits)} "
                    f"qubit(s) and {len(cbits)} bit(s). If you think this "
                    f"should be supported and is a bug, please open an issue "
                    f"at https://github.com/unitaryfund/mitiq."
                )
            order.append((*qubits, *cbits))
    return order


def _transform_registers(
    circuit: qiskit.QuantumCircuit,
    new_qregs: Optional[List[qiskit.QuantumRegister]] = None,
    new_cregs: Optional[List[qiskit.ClassicalRegister]] = None,
) -> None:
    """Transforms the registers in the circuit to the new registers.

    Args:
        circuit: Qiskit circuit with one quantum register and either
            * No classical registers, or
            * One single classical register of n bits, or
            * n single-bit classical registers.
        new_qregs: The new quantum registers for the circuit.
        new_cregs: The new classical registers for the circuit.

    Raises:
        ValueError:
            * If the input circuit has more than one quantum register.
            * If the number of qubits in the new quantum registers does not
            match the number of qubits in the circuit.
            * If the input circuit has a classical register with more than one
            bit.
            * If the number of bits in the new classical registers does not
            match the number of bits in the circuit.
    """
    if new_qregs is None and new_cregs is None:
        return

    if new_qregs is None:
        new_qregs = []

    if new_cregs is None:
        new_cregs = []

    qreg_sizes = [qreg.size for qreg in new_qregs]
    old_qregs = circuit.qregs
    nqubits_in_circuit = sum(qreg.size for qreg in old_qregs)

    if len(old_qregs) > 1:
        raise ValueError(
            "Input circuit is required to have <= 1 quantum register but has "
            f"{len(circuit.qregs)} quantum registers."
        )

    if len(qreg_sizes) and sum(qreg_sizes) != nqubits_in_circuit:
        raise ValueError(
            f"The circuit has {nqubits_in_circuit} qubits, but the provided "
            f"quantum registers have {sum(qreg_sizes)} qubits."
        )

    creg_sizes = [creg.size for creg in new_cregs]
    old_cregs = circuit.cregs
    nbits_in_circuit = sum(creg.size for creg in old_cregs)

    if len(old_cregs) not in (0, 1, nbits_in_circuit):
        raise ValueError(
            f"Input circuit is required to have 0, 1, or {nbits_in_circuit} "
            f"classical registers but has {len(circuit.cregs)} classical "
            f"registers."
        )

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

        new_qubits = _map_bits(qubits, old_qregs, qreg_sizes, new_qregs)
        new_cbits = _map_bits(cbits, old_cregs, creg_sizes, new_cregs)

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
    add_cregs_if_cannot_transform: bool = True,
) -> qiskit.QuantumCircuit:
    """Returns a Qiskit circuit equivalent to the input Mitiq circuit.

    Args:
        circuit: Mitiq circuit to convert to a Qiskit circuit.
        qregs: Quantum registers of the returned Qiskit circuit. If none are
            provided, a single default register is used.
        cregs: Classical registers of the returned Qiskit circuit, provided
            that the original circuit has classical registers and
            ``add_cregs_if_cannot_transform`` is True. If none are provided, a
            single default register is used.
        add_cregs_if_cannot_transform: If True, the provided ``cregs`` are
            added to the circuit if there are no classical registers in the
            original ``circuit``.

    Returns:
        Qiskit.QuantumCircuit object equivalent to the input Mitiq circuit.
    """
    # Base conversion.
    qiskit_circuit = qiskit.QuantumCircuit.from_qasm_str(to_qasm(circuit))

    # Assign register structure.
    # Note: Output qiskit_circuit has one quantum register and n classical
    # registers of 1 bit where n is the total number of classical bits.
    if len(qiskit_circuit.cregs) > 0:
        _transform_registers(qiskit_circuit, new_qregs=qregs, new_cregs=cregs)
    else:
        _transform_registers(qiskit_circuit, new_qregs=qregs)
        if cregs and add_cregs_if_cannot_transform:
            qiskit_circuit.add_register(*cregs)

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
