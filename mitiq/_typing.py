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

"""This file is used to optionally import what program types should be allowed
by mitiq based on what packages are installed in the environment.

Supported program types:
    cirq.Circuit
    pyquil.Program
    qiskit.QuantumCircuit

When pyquil or qiskit are not available for import, we take advantage of a
clever hack in the except ImportError block, setting the type alias
corresponding to the class that we failed to import equal to cirq.Circuit,
which is then handled gracefully by the Union type.
"""
from typing import Union

from cirq import Circuit as _Circuit

try:
    from pyquil import Program as _Program
except ImportError:  # pragma: no cover
    _Program = _Circuit

try:
    from qiskit import QuantumCircuit as _QuantumCircuit
except ImportError:  # pragma: no cover
    _QuantumCircuit = _Circuit

QPROGRAM = Union[_Circuit, _Program, _QuantumCircuit]

SUPPORTED_PROGRAM_TYPES = {
    "cirq": "Circuit",
    "pyquil": "Program",
    "qiskit": "QuantumCircuit",
}
