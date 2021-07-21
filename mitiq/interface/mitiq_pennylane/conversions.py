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
from cirq.contrib.qasm_import import circuit_from_qasm
from pennylane_qiskit import AerDevice, load_qasm


def from_pennylane(device: AerDevice) -> Circuit:
    return circuit_from_qasm(device._circuit.qasm())


def to_pennylane(circuit: Circuit):
    return load_qasm(circuit.to_qasm())
