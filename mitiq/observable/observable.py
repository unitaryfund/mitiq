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
import cirq

from mitiq.observable.pauli import PauliString, PauliStringSet
from mitiq._typing import MeasurementResult, QuantumResult, QPROGRAM
from mitiq.collector import Collector


class Observable:
    def __init__(self, *paulis: PauliString) -> None:
        # TODO: Add option to Combine duplicates. E.g. [Z(0, Z(0)] -> [2*Z(0)].
        self._paulis = list(paulis)
        self._groups: List[PauliStringSet]
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
    def groups(self) -> List[PauliStringSet]:
        return self._groups

    @property
    def ngroups(self) -> int:
        return self._ngroups

    def partition(self, seed: Optional[int] = None) -> None:
        rng = np.random.RandomState(seed)

        psets: List[PauliStringSet] = []
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
                psets.append(PauliStringSet(pauli))

        self._groups = psets
        self._ngroups = len(self._groups)

    def _measure_in(self, circuit: cirq.Circuit) -> List[cirq.Circuit]:
        return [pset.measure_in(circuit) for pset in self._groups]

    def matrix(self, dtype: type = np.complex128) -> np.ndarray:
        """Returns the (potentially very large) matrix of the Observable."""
        qubit_indices = self.qubit_indices
        n = self.nqubits

        matrix = np.zeros(shape=(2 ** n, 2 ** n), dtype=dtype)
        for pauli in self._paulis:
            matrix += pauli.matrix(
                qubit_indices_to_include=qubit_indices
            ).astype(dtype=dtype)

        return matrix

    def expectation(
        self, circuit: QPROGRAM, executor: Callable[[QPROGRAM], QuantumResult]
    ) -> float:
        collector = Collector(executor=executor)
        to_run = self._measure_in(circuit=circuit)
        print("IN EXPECTATION, circuits to run are:")
        for c in to_run:
            print(c)
        quantum_results = collector.run(circuits=to_run)
        print("JUST RAN, results are:")
        print(quantum_results)
        return self.expectation_from(quantum_results=quantum_results)

    def expectation_from(self, quantum_results: List[QuantumResult]) -> float:
        # TODO: Dispatch to correct function based on type of quantum result.
        result = quantum_results[0]
        # if isinstance(result, MeasurementResult):
        return self._expectation_from_measurements(quantum_results)
        # else:
        #     raise NotImplementedError

    def _expectation_from_wavefunction(
        self, wavefunction: np.ndarray
    ) -> float:
        return wavefunction.conj().T @ self.matrix() @ wavefunction

    def _expectation_from_trajectories(
        self, wavefunctions: List[np.ndarray]
    ) -> float:
        return np.sum(
            [self._expectation_from_wavefunction(wf) for wf in wavefunctions]
        ) / len(wavefunctions)

    def _expectation_from_density_matrix(
        self, density_matrix: np.ndarray
    ) -> float:
        return np.trace(density_matrix @ self.matrix()).real

    def _expectation_from_measurements(
        self, measurements: List[MeasurementResult]
    ) -> float:
        expectation = 0.0

        for (group, bitstrings) in zip(self._groups, measurements):
            for pauli in group:
                expectation += pauli._expectation_from_measurements(bitstrings)

        return expectation
