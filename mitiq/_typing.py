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

"""Defines input / output types for a quantum computer (simulator):

  * SUPPORTED_PROGRAM_TYPES: All supported packages / circuits which Mitiq can
       interface with,
  * QPROGRAM: All supported packages / circuits which are installed in the
       environment Mitiq is run in, and
  * QuantumResult: An object returned by a quantum computer (simulator) running
       a quantum program from which expectation values to be mitigated can be
       computed. Note this includes expectation values themselves.
"""
from typing import Union

import numpy as np

from cirq import Circuit as _Circuit
from mitiq.rem.measurement_result import MeasurementResult


# Supported quantum programs.
SUPPORTED_PROGRAM_TYPES = {
    "cirq": "Circuit",
    "pyquil": "Program",
    "qiskit": "QuantumCircuit",
    "braket": "Circuit",
}


try:
    from pyquil import Program as _Program
except ImportError:  # pragma: no cover
    _Program = _Circuit  # type: ignore

try:
    from qiskit import QuantumCircuit as _QuantumCircuit
except ImportError:  # pragma: no cover
    _QuantumCircuit = _Circuit

try:
    from braket.circuits import Circuit as _BKCircuit
except ImportError:  # pragma: no cover
    _BKCircuit = _Circuit

# Supported + installed quantum programs.
QPROGRAM = Union[_Circuit, _Program, _QuantumCircuit, _BKCircuit]


# An `executor` function inputs a quantum program and outputs an object from
# which expectation values can be computed. Explicitly, this object can be one
# of the following types:
QuantumResult = Union[
    float,  # The expectation value itself.
    MeasurementResult,  # Sampled bitstrings.
    np.ndarray,  # Density matrix.
    # TODO: Support the following:
    # Sequence[np.ndarray],  # Wavefunctions sampled via quantum trajectories.
]
