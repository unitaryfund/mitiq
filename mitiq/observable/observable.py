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

import copy
from typing import Callable, cast, List, Optional, Set

import numpy as np
import numpy.typing as npt
import cirq

from mitiq.observable.pauli import PauliString, PauliStringCollection
from mitiq._typing import MeasurementResult, QuantumResult, QPROGRAM


class Observable:
    """A quantum observable typically used to compute its mitigated expectation
    value.

    """

    def __init__(self, *paulis: PauliString) -> None:
        """Initializes an `Observable` with :class:`.PauliString` objects.

        Args:
            paulis: PauliStrings used to define the observable.

        """
        # TODO: Add option to Combine duplicates. E.g. [Z(0, Z(0)] -> [2*Z(0)].
        self._paulis = list(paulis)
        self._groups: List[PauliStringCollection]
        self._ngroups: int
        self.partition()

    @staticmethod
    def from_pauli_string_collections(
        *pauli_string_collections: PauliStringCollection,
    ) -> "Observable":
        obs = Observable()
        obs._groups = list(pauli_string_collections)
        obs._ngroups = len(pauli_string_collections)
        obs._paulis = [
            pauli
            for pauli_string_collection in pauli_string_collections
            for pauli in pauli_string_collection.elements
        ]
        return obs

    @property
    def nterms(self) -> int:
        return len(self._paulis)

    def _qubits(self) -> Set[cirq.Qid]:
        """Returns all qubits acted on by the Observable."""
        return {q for pauli in self._paulis for q in pauli._pauli.qubits}

    @property
    def qubit_indices(self) -> List[int]:
        return [cast(cirq.LineQubit, q).x for q in sorted(self._qubits())]

    @property
    def nqubits(self) -> int:
        return len(self.qubit_indices)

    @property
    def groups(self) -> List[PauliStringCollection]:
        return self._groups

    @property
    def ngroups(self) -> int:
        return self._ngroups

    def partition(self, seed: Optional[int] = None) -> None:
        rng = np.random.RandomState(seed)

        psets: List[PauliStringCollection] = []
        paulis = copy.deepcopy(self._paulis)
        rng.shuffle(paulis)  # type: ignore

        while paulis:
            pauli = paulis.pop()
            added = False
            for (i, pset) in enumerate(psets):
                if pset.can_add(pauli):
                    pset.add(pauli)
                    added = True
                    break

            if not added:
                psets.append(PauliStringCollection(pauli))

        self._groups = psets
        self._ngroups = len(self._groups)

    def measure_in(self, circuit: QPROGRAM) -> List[QPROGRAM]:
        return [pset.measure_in(circuit) for pset in self._groups]

    def matrix(
        self,
        qubit_indices: Optional[List[int]] = None,
    ) -> npt.NDArray[np.complex64]:
        """Returns the (potentially very large) matrix of the Observable."""
        if qubit_indices is None:
            qubit_indices = self.qubit_indices
        n = len(qubit_indices)

        obs_matrix = np.zeros(shape=(2**n, 2**n), dtype=np.complex64)
        for pauli in self._paulis:
            obs_matrix += pauli.matrix(qubit_indices_to_include=qubit_indices)

        return obs_matrix

    def expectation(
        self, circuit: QPROGRAM, execute: Callable[[QPROGRAM], QuantumResult]
    ) -> complex:
        from mitiq.executor import Executor  # Avoid circular import.

        return Executor(execute).evaluate(circuit, observable=self)[0]

    def _expectation_from_measurements(
        self, measurements: List[MeasurementResult]
    ) -> float:
        return sum(
            pset._expectation_from_measurements(bitstrings)
            for (pset, bitstrings) in zip(self._groups, measurements)
        )

    def _expectation_from_density_matrix(
        self, density_matrix: npt.NDArray[np.complex64]
    ) -> float:
        observable_matrix = self.matrix()

        if density_matrix.shape != observable_matrix.shape:
            nqubits = int(np.log2(density_matrix.shape[0]))
            density_matrix = cirq.partial_trace(
                np.reshape(density_matrix, newshape=[2, 2] * nqubits),
                keep_indices=self.qubit_indices,
            ).reshape(observable_matrix.shape)

        return np.real_if_close(
            np.trace(density_matrix @ observable_matrix)
        ).item()

    def __str__(self) -> str:
        return " + ".join(map(str, self._paulis))
