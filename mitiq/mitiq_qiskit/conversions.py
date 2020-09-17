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
import cirq
from cirq.contrib.qasm_import import circuit_from_qasm
from qiskit import QuantumCircuit
from qiskit.extensions import Barrier

from mitiq.utils import _simplify_circuit_exponents


QASMType = str


def _remove_barriers(circuit: QuantumCircuit) -> QuantumCircuit:
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


def to_qiskit(circuit: cirq.Circuit) -> QuantumCircuit:
    """Returns a Qiskit circuit equivalent to the input Mitiq circuit.

    Args:
        circuit: Mitiq circuit to convert to a Qiskit circuit.

    Returns:
        Qiskit.QuantumCircuit object equivalent to the input Mitiq circuit.
    """
    return QuantumCircuit.from_qasm_str(to_qasm(circuit))


def from_qiskit(circuit: QuantumCircuit) -> cirq.Circuit:
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
    qasm = _remove_barriers(QuantumCircuit.from_qasm_str(qasm)).qasm()
    return circuit_from_qasm(qasm)
