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
import inspect
from typing import Callable, cast, List, Optional, Set

import numpy as np
import cirq

from mitiq.observable.pauli import PauliString, PauliStringCollection
from mitiq._typing import MeasurementResult, QuantumResult, QPROGRAM
from mitiq.executor import Executor


class Observable:
    def __init__(self, *paulis: PauliString) -> None:
        # TODO: Add option to Combine duplicates. E.g. [Z(0, Z(0)] -> [2*Z(0)].
        self._paulis = list(paulis)
        self._groups: List[PauliStringCollection]
        self._ngroups: int
        self.partition()

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
        rng.shuffle(paulis)

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
        dtype: type = np.complex128,
    ) -> np.ndarray:
        """Returns the (potentially very large) matrix of the Observable."""
        if qubit_indices is None:
            qubit_indices = self.qubit_indices
        n = len(qubit_indices)

        matrix = np.zeros(shape=(2 ** n, 2 ** n), dtype=dtype)
        for pauli in self._paulis:
            matrix += pauli.matrix(
                qubit_indices_to_include=qubit_indices
            ).astype(dtype=dtype)

        return matrix

    def expectation(
        self, circuit: QPROGRAM, execute: Callable[[QPROGRAM], QuantumResult]
    ) -> float:
        result_type = inspect.getfullargspec(execute).annotations.get("return")

        if result_type is MeasurementResult:
            to_run = self.measure_in(circuit)
            results = Executor(execute).run(to_run)
            return self._expectation_from_measurements(
                cast(List[MeasurementResult], results)
            )
        elif result_type is np.ndarray:
            density_matrix = cast(np.ndarray, execute(circuit))
            observable_matrix = self.matrix()

            if density_matrix.shape != observable_matrix.shape:
                nqubits = int(np.log2(density_matrix.shape[0]))
                density_matrix = cirq.partial_trace(
                    np.reshape(density_matrix, newshape=[2, 2] * nqubits),
                    keep_indices=self.qubit_indices,
                ).reshape(observable_matrix.shape)

            return np.trace(density_matrix @ self.matrix())
        else:
            raise ValueError(
                f"Arg `execute` must be a function with annotated return type "
                f"that is either mitiq.MeasurementResult or np.ndarray but "
                f"was {result_type}."
            )

    def _expectation_from_measurements(
        self, measurements: List[MeasurementResult]
    ) -> float:
        return sum(
            pset._expectation_from_measurements(bitstrings)
            for (pset, bitstrings) in zip(self._groups, measurements)
        )
