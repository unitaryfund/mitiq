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

"""Defines MeasurementResult, a result obtained by measuring qubits on a
quantum computer."""
from dataclasses import dataclass
from typing import cast, Iterable, List, Optional, Tuple

import numpy as np


Bitstring = List[int]


@dataclass
class MeasurementResult:
    """Bitstrings sampled from a quantum computer."""

    result: List[Bitstring]
    qubit_indices: Optional[Tuple[int, ...]] = None

    def __post_init__(self) -> None:
        if not set(b for bits in self.result for b in bits).issubset({0, 1}):
            raise ValueError(
                "MeasurementResult contains elements which are not (0, 1)."
            )

        self._bitstrings = np.array(self.result)
        if isinstance(self.result, np.ndarray):
            self.result = cast(List[Bitstring], self.result.tolist())

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
    def asarray(self) -> np.ndarray:
        return self._bitstrings

    def __getitem__(self, indices: List[int]) -> np.ndarray:
        return np.array([self._measurements[i] for i in indices]).T

    def __iter__(self) -> Iterable[Bitstring]:
        yield from self.result
