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
"""
from typing import Union

SUPPORTED_PROGRAM_TYPES = {
    "cirq": "Circuit",
    "pyquil": "Program",
    "qiskit": "QuantumCircuit",
}

AVAILABLE_PROGRAM_TYPES = {}

for (module, program_type) in SUPPORTED_PROGRAM_TYPES.items():
    try:
        exec(f"from {module} import {program_type}")
        AVAILABLE_PROGRAM_TYPES.update({module: program_type})
    except ImportError:
        pass

QPROGRAM = Union[
    tuple(
        f"{package}.{circuit}"
        for package, circuit in AVAILABLE_PROGRAM_TYPES.items()
    )
]
