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
pyQuil's circuit representation (Quil programs).
"""
from cirq import Circuit, LineQubit
from cirq_rigetti import circuit_from_quil
from cirq_rigetti.quil_output import QuilOutput
from pyquil import Program

QuilType = str


def to_quil(circuit: Circuit) -> QuilType:
    """Returns a Quil string representing the input Mitiq circuit.

    Args:
        circuit: Mitiq circuit to convert to a Quil string.

    Returns:
        QuilType: Quil string equivalent to the input Mitiq circuit.
    """
    max_qubit = max(circuit.all_qubits())
    # if we are using LineQubits, keep the qubit labeling the same
    if isinstance(max_qubit, LineQubit):
        qubit_range = max_qubit.x + 1
        return str(
            QuilOutput(circuit.all_operations(), LineQubit.range(qubit_range))
        )
    # otherwise, use the default ordering (starting from zero)
    return str(
        QuilOutput(circuit.all_operations(), sorted(circuit.all_qubits()))
    )


def to_pyquil(circuit: Circuit) -> Program:
    """Returns a pyQuil Program equivalent to the input Mitiq circuit.

    Args:
        circuit: Mitiq circuit to convert to a pyQuil Program.

    Returns:
        pyquil.Program object equivalent to the input Mitiq circuit.
    """
    return Program(to_quil(circuit))


def from_pyquil(program: Program) -> Circuit:
    """Returns a Mitiq circuit equivalent to the input pyQuil Program.

    Args:
        program: PyQuil Program to convert to a Mitiq circuit.

    Returns:
        Mitiq circuit representation equivalent to the input pyQuil Program.
    """
    return from_quil(program.out())


def from_quil(quil: QuilType) -> Circuit:
    """Returns a Mitiq circuit equivalent to the input Quil string.

    Args:
        quil: Quil string to convert to a Mitiq circuit.

    Returns:
        Mitiq circuit representation equivalent to the input Quil string.
    """
    return circuit_from_quil(quil)
