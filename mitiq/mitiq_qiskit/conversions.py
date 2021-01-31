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

from typing import List, Tuple

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


def _map_qubit_index(
    qubit_index: int, new_register_sizes: List[int]
) -> Tuple[int, int]:
    """Returns the register index and qubit index in this register for the
    mapped qubit_index.

    Args:
        qubit_index: Index of qubit in the original register.
        new_register_sizes: List of sizes of the new registers.

    Example:
        qubit_index = 3, new_register_sizes = [2, 3]
        returns (1, 0), meaning the mapped qubit is in the 1st new register and
        has index 0 in this register.
    """
    max_indices_in_registers = np.cumsum(new_register_sizes) - 1

    # Could be faster via bisection.
    register_index = None
    for i in range(len(max_indices_in_registers)):
        if qubit_index <= max_indices_in_registers[i]:
            register_index = i
            break
    assert register_index is not None

    if register_index == 0:
        return register_index, qubit_index

    return (
        register_index,
        qubit_index - max_indices_in_registers[register_index - 1] - 1,
    )


def _transform_quantum_registers(
    circuit: qiskit.QuantumCircuit, *new_registers: qiskit.QuantumRegister
) -> None:
    """Transforms the quantum registers in the circuit to the new registers.

    Args:
        circuit: Qiskit circuit with one quantum register.
        new_registers: The quantum registers the circuit will act on.

    Raises:
        ValueError:
            * If the input circuit has more than one quantum register.
            * If the number of qubits in the new registers does not match the
            number of qubits in the circuit.
    """
    if not len(new_registers):
        return

    if len(circuit.qregs) != 1:
        raise ValueError(
            "Input circuit is required to have 1 quantum register but has "
            f"{len(circuit.qregs)} quantum registers."
        )

    register_sizes = [qreg.size for qreg in new_registers]
    nqubits_in_circuit = sum(qreg.size for qreg in circuit.qregs)

    if sum(register_sizes) != nqubits_in_circuit:
        raise ValueError(
            f"The circuit has {nqubits_in_circuit} qubits, but the provided "
            f"registers have {sum(register_sizes)} qubits."
        )

    circuit.qregs = list(new_registers)
    new_ops = []
    for op in circuit.data:
        gate, qubits, cbits = op
        qubit_indices = [qubit.index for qubit in qubits]
        mapped_indices = [
            _map_qubit_index(index, register_sizes)
            for index in qubit_indices
        ]
        new_qubits = [
            qiskit.circuit.quantumregister.Qubit(new_registers[i], j)
            for i, j in mapped_indices
        ]
        new_ops.append((gate, new_qubits, cbits))

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
    circuit: cirq.Circuit, *registers: qiskit.QuantumRegister
) -> qiskit.QuantumCircuit:
    """Returns a Qiskit circuit equivalent to the input Mitiq circuit.

    Args:
        circuit: Mitiq circuit to convert to a Qiskit circuit.
        registers: Quantum registers of the returned Qiskit circuit. If none
            are provided, a single default register is used.

    Returns:
        Qiskit.QuantumCircuit object equivalent to the input Mitiq circuit.
    """
    # Base conversion.
    qiskit_circuit = qiskit.QuantumCircuit.from_qasm_str(to_qasm(circuit))

    # Assign register structure.
    _transform_quantum_registers(qiskit_circuit, *registers)

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
