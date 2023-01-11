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
from dataclasses import dataclass
from typing import cast, Iterable, List, Optional, Tuple, Union, Sequence, Dict
from typing import Counter as CounterType

import numpy as np
import numpy.typing as npt

from cirq import Circuit as _Circuit

from collections import Counter


# Supported quantum programs.
SUPPORTED_PROGRAM_TYPES = {
    "cirq": "Circuit",
    "pyquil": "Program",
    "qiskit": "QuantumCircuit",
    "braket": "Circuit",
    "pennylane": "QuantumTape",
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

try:
    from pennylane.tape import QuantumTape as _QuantumTape
except ImportError:  # pragma: no cover
    _QuantumTape = _Circuit


# Supported + installed quantum programs.
QPROGRAM = Union[_Circuit, _Program, _QuantumCircuit, _BKCircuit, _QuantumTape]


# Define MeasurementResult, a result obtained by measuring qubits on a quantum
# computer.
Bitstring = Union[str, List[int]]


@dataclass
class MeasurementResult:
    """Bitstrings sampled from a quantum computer."""

    result: Sequence[Bitstring]
    qubit_indices: Optional[Tuple[int, ...]] = None

    def __post_init__(self) -> None:
        # Validate arguments
        symbols = set(b for bits in self.result for b in bits)
        if not (symbols.issubset({0, 1}) or symbols.issubset({"0", "1"})):
            raise ValueError("Bitstrings should look like '011' or [0, 1, 1].")

        if symbols.issubset({"0", "1"}):
            # Convert to list of integers
            int_result = [[int(b) for b in bits] for bits in self.result]
            self.result: List[List[int]] = list(int_result)

        if isinstance(self.result, np.ndarray):
            self.result = cast(List[Bitstring], self.result.tolist())

        self._bitstrings = np.array(self.result)

        if not self.qubit_indices:
            self.qubit_indices = tuple(range(self.nqubits))
        else:
            if len(self.qubit_indices) != self.nqubits:
                raise ValueError(
                    f"MeasurementResult has {self.nqubits} qubit(s) but there "
                    f"are {len(self.qubit_indices)} `qubit_indices`."
                )

        self._measurements = dict(zip(self.qubit_indices, self._bitstrings.T))

    @property
    def shots(self) -> int:
        return self._bitstrings.shape[0]

    @property
    def nqubits(self) -> int:
        return (
            self._bitstrings.shape[1]
            if len(self._bitstrings.shape) >= 2
            else 0
        )

    @property
    def asarray(self) -> npt.NDArray[np.int64]:
        return self._bitstrings

    @classmethod
    def from_dict(
        cls,
        counts: Sequence[Bitstring],
        qubit_indices: Optional[Tuple[int, ...]] = None,
    ) -> "MeasurementResult":
        """Initializes a MeasurementResult from a dictionary of counts.

        Example:
        >>> MeasurementResult.from_dict({"00": 175, "11": 177})
        """
        counter = Counter(counts)
        return cls(list(counter.elements()), qubit_indices)

    def to_counter(self) -> CounterType[str]:
        """Returns a colleciton.Counter whose keys are the measured
        bitstrings the and whose values are the counts.

        Note: Qubit indices (self.qubit_indeces) are lost in the conversion.
        """
        strings = ["".join(map(str, bits)) for bits in self.result]
        print(strings)
        return Counter(strings)

    def to_dict(self) -> Dict[str, int]:
        """Returns a Python dictionary whose keys are the measured
        bitstrings the and whose values are the counts.

        Note: Qubit indices (self.qubit_indeces) are lost in the conversion.
        """
        return {**self.to_counter()}

    def __getitem__(self, indices: List[int]) -> npt.NDArray[np.int64]:
        return np.array([self._measurements[i] for i in indices]).T

    def __iter__(self) -> Iterable[Bitstring]:
        yield from self.result


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
