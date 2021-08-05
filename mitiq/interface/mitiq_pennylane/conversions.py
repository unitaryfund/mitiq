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

"""Functions to convert between Mitiq's internal circuit representation and
Pennylane's circuit representation.
"""

from cirq import Circuit
from mitiq.interface.mitiq_qiskit import from_qasm as cirq_from_qasm, to_qasm
from pennylane.wires import Wires
from pennylane.tape import QuantumTape

from pennylane import from_qasm as pennylane_from_qasm


class UnsupportedQuantumTapeError(Exception):
    pass


def from_pennylane(tape: QuantumTape) -> Circuit:
    """Returns a Cirq circuit equivalent to the input QuantumTape.

    Args:
        tape: Pennylane QuantumTape to convert to a Cirq circuit.
    """
    try:
        wires = sorted(tape.wires)
    except TypeError:
        raise UnsupportedQuantumTapeError(
            f"The wires of the tape must be sortable, but could not sort "
            f"{tape.wires}."
        )

    return cirq_from_qasm(tape.to_openqasm(rotations=True, wires=wires))


def to_pennylane(circuit: Circuit) -> QuantumTape:
    """Returns a QuantumTape equivalent to the input Cirq circuit.

    Args:
        circuit: Cirq circuit to convert to a Pennylane QuantumTape.
    """
    qfunc = pennylane_from_qasm(to_qasm(circuit))

    with QuantumTape() as tape:
        qfunc(wires=Wires(range(len(circuit.all_qubits()))))

    return tape
