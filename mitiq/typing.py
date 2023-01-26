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
from typing import cast, List, Optional, Tuple, Union, Sequence, Dict, Any

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
    """Mitiq object for collecting the bitstrings sampled from a quantum
    computer when executing a circuit. This is one of the possible types
    (see :class:`~mitiq.typing.QuantumResult`) that an
    :class:`.Executor` can return.

    Args:
        result:
            The sequence of measured bitstrings.
        qubit_indices:
            The qubit indices associated to each
            bit in a bitstring (from left to right).
            If not given, Mitiq assumes the default ordering
            ``tuple(range(self.nqubits))``, where ``self.nqubits``
            is the bitstring length deduced from ``result``.

    Note:
        Use caution when selecting the default option for ``qubit_indices``,
        especially when estimating an :class:`.Observable`
        acting on a subset of qubits. In this case Mitiq
        only applies measurement gates to the specific qubits and, therefore,
        it is essential to specify the corresponding ``qubit_indices``.
    """

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
    def from_counts(
        cls,
        counts: Dict[str, int],
        qubit_indices: Optional[Tuple[int, ...]] = None,
    ) -> "MeasurementResult":
        """Initializes a ``MeasurementResult`` from a dictionary of counts.

        **Example**::

            MeasurementResult.from_counts({"00": 175, "11": 177})
        """
        counter = Counter(counts)
        return cls(list(counter.elements()), qubit_indices)

    def get_counts(self) -> Dict[str, int]:
        """Returns a Python dictionary whose keys are the measured
        bitstrings and whose values are the counts.
        """
        strings = ["".join(map(str, bits)) for bits in self.result]
        return dict(Counter(strings))

    def prob_distribution(self) -> Dict[str, float]:
        """Returns a Python dictionary whose keys are the measured
        bitstrings and whose values are their empirical frequencies.
        """
        return {k: v / self.shots for k, v in self.get_counts().items()}

    def to_dict(self) -> Dict[str, Any]:
        """Exports data to a Python dictionary.

        Note: Information about the order measurements is not preserved.
        """

        return {
            "nqubits": self.nqubits,
            "qubit_indices": self.qubit_indices,
            "shots": self.shots,
            "counts": self.get_counts(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MeasurementResult":
        """Loads a ``MeasurementResult`` from a Python dictionary.

        Note: Only ``data["counts"]`` and ``data["qubit_indices"]`` are used
        by this method. Total shots and number of qubits are deduced.
        """
        return cls.from_counts(data["counts"], data["qubit_indices"])

    def filter_qubits(self, qubit_indices: List[int]) -> npt.NDArray[np.int64]:
        """Returns the bitstrings associated to a subset of qubits."""
        return np.array([self._measurements[i] for i in qubit_indices]).T

    def __repr__(self) -> str:
        # We redefine __repr__ in this way to avoid very long output strings.
        return "MeasurementResult: " + str(self.to_dict())


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
