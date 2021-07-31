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

from typing import Optional, List

from cirq import Circuit
from mitiq.interface.mitiq_qiskit import from_qasm as cirq_from_qasm, to_qasm
from pennylane.wires import Wires
from pennylane.measure import MeasurementProcess
from pennylane.tape import QuantumTape

from pennylane import from_qasm as pennylane_from_qasm


def from_pennylane(circuit: QuantumTape) -> Circuit:
    """Returns a Cirq circuit equivalent to the input QuantumTape.

    Args:
        circuit: Pennylane QuantumTape to convert to a Cirq circuit.
    """
    return cirq_from_qasm(circuit.to_openqasm(rotations=False))


def to_pennylane(
    circuit: Circuit,
    target_wires: Optional[Wires] = None,
) -> QuantumTape:
    """Returns a QuantumTape equivalent to the input Cirq circuit.

    Args:
        circuit: Cirq circuit to convert to a Pennylane QuantumTape.
    """
    if target_wires is None:
        target_wires = Wires(range(len(circuit.all_qubits())))

    qfunc = pennylane_from_qasm(to_qasm(circuit))

    with QuantumTape() as tape:
        qfunc(wires=target_wires)

    return tape
