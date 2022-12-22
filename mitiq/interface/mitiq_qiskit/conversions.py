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
import copy
from typing import List, Optional, Tuple, Any, Set
import re

import numpy as np

import cirq
from cirq.contrib.qasm_import import circuit_from_qasm
import qiskit

from mitiq.utils import _simplify_circuit_exponents


QASMType = str


def _remove_qasm_barriers(qasm: QASMType) -> QASMType:
    """Returns a copy of the input QASM with all barriers removed.

    Args:
        qasm: QASM to remove barriers from.

    Note:
        According to the OpenQASM 2.X language specification
        (https://arxiv.org/pdf/1707.03429v2.pdf), "Statements are separated by
        semicolons. Whitespace is ignored. The language is case sensitive.
        Comments begin with a pair of forward slashes and end with a new line."
    """
    quoted_re = r"(?:\"[^\"]*?\")"
    statement_re = r"((?:[^;{}\"]*?" + quoted_re + r"?)*[;{}])?"
    comment_re = r"(\n?//[^\n]*(?:\n|$))?"
    statements_comments = re.findall(statement_re + comment_re, qasm)
    lines = []
    for statement, comment in statements_comments:
        if re.match(r"^\s*barrier(?:(?:\s+)|(?:;))", statement) is None:
            lines.append(statement + comment)
    return "".join(lines)


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


def _map_qubits(
    qubits: List[qiskit.circuit.Qubit],
    new_register_sizes: List[int],
    new_registers: List[qiskit.QuantumRegister],
) -> List[qiskit.circuit.Qubit]:
    """Maps qubits to new registers.

    Args:
        qubits: A list of qubits to map.
        new_register_sizes: The size(s) of the new registers to map to.
            Note: These can be determined from ``new_registers``, but this
            helper function is only called from ``_transform_registers`` where
            the sizes are already computed.
        new_registers: The new registers to map the ``qubits`` to.

    Returns:
        The input ``qubits`` mapped to the ``new_registers``.
    """
    indices = [bit.index for bit in qubits]
    mapped_indices = [_map_bit_index(i, new_register_sizes) for i in indices]
    return [
        qiskit.circuit.Qubit(new_registers[i], j) for i, j in mapped_indices
    ]


def _add_identity_to_idle(
    circuit: qiskit.QuantumCircuit,
) -> Set[qiskit.circuit.Qubit]:
    """Adds identities to idle qubits in the circuit and returns the altered
    indices. Used to preserve idle qubits and indices in conversion.

    Args:
        circuit: Qiskit circuit to have identities added to idle qubits

    Returns:
        An unordered set of the indices that were altered

    Note: An idle qubit is a qubit without any gates (including Qiskit
        barriers) acting on it.
    """
    all_qubits = set(circuit.qubits)
    used_qubits = set()
    idle_qubits = set()
    # Get used qubits
    for op in circuit.data:
        _, qubits, _ = op
        used_qubits.update(set(qubits))
    idle_qubits = all_qubits - used_qubits
    # Modify input circuit applying I to idle qubits
    for q in idle_qubits:
        circuit.i(q)

    return idle_qubits


def _remove_identity_from_idle(
    circuit: qiskit.QuantumCircuit,
    idle_qubits: Set[qiskit.circuit.Qubit],
) -> None:
    """Removes identities from the circuit corresponding to the input
    idle qubits.
    Used in conjunction with _add_identity_to_idle to preserve idle qubits in
    conversion.

    Args:
        circuit: Qiskit circuit to have identities removed
        idle_indices: Set of altered idle qubits.
    """
    to_delete_indices: List[int] = []
    for index, op in enumerate(circuit._data):
        gate, qubits, cbits = op
        if gate.name == "id" and set(qubits).intersection(idle_qubits):
            to_delete_indices.append(index)
    # Traverse data from list end to preserve index
    for index in to_delete_indices[::-1]:
        del circuit._data[index]


def _measurement_order(
    circuit: qiskit.QuantumCircuit,
) -> List[Tuple[Any, ...]]:
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
) -> None:
    """Transforms the registers in the circuit to the new registers.

    Args:
        circuit: Qiskit circuit with at most one quantum register.
        new_qregs: The new quantum registers for the circuit.

    Raises:
        ValueError:
            * If the input circuit has more than one quantum register.
            * If the number of qubits in the new quantum registers is
            greater than the number of qubits in the circuit.
    """
    if new_qregs is None:
        return

    if len(circuit.qregs) > 1:
        raise ValueError(
            "Input circuit is required to have <= 1 quantum register but has "
            f"{len(circuit.qregs)} quantum registers."
        )

    qreg_sizes = [qreg.size for qreg in new_qregs]
    nqubits_in_circuit = circuit.num_qubits

    if len(qreg_sizes) and sum(qreg_sizes) < nqubits_in_circuit:
        raise ValueError(
            f"The circuit has {nqubits_in_circuit} qubit(s), but the provided "
            f"quantum registers have {sum(qreg_sizes)} qubit(s)."
        )

    # Copy the circuit data.
    data = copy.deepcopy(circuit._data)

    # Remove the old qubits and add the new ones.
    circuit._qubits = []
    circuit._qubit_set = set()
    circuit.qregs = []
    circuit._data = []
    circuit._qubit_indices = {}
    circuit.add_register(*new_qregs)

    # Map the qubits in operations to the new qubits.
    for op in data:
        gate, qubits, cbits = op
        new_qubits = _map_qubits(qubits, qreg_sizes, new_qregs)
        circuit.append(gate, new_qubits, cbits)


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


def to_qiskit(circuit: cirq.Circuit) -> qiskit.QuantumCircuit:
    """Returns a Qiskit circuit equivalent to the input Mitiq circuit. Note
    that the output circuit registers may not match the input circuit
    registers.

    Args:
        circuit: Mitiq circuit to convert to a Qiskit circuit.

    Returns:
        Qiskit.QuantumCircuit object equivalent to the input Mitiq circuit.
    """
    return qiskit.QuantumCircuit.from_qasm_str(to_qasm(circuit))


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
    qasm = _remove_qasm_barriers(qasm)
    return circuit_from_qasm(qasm)
