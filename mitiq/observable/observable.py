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
from typing import Callable, cast, FrozenSet, List, Set

import numpy as np
import cirq

from mitiq.observable.pauli import PauliString
from mitiq._typing import MeasurementResult, QuantumResult, QPROGRAM
from mitiq.collector import Collector


class Observable:
    def __init__(self, *paulis: PauliString) -> None:
        self._paulis = set(paulis)
        self._nterms = len(self._paulis)
        self.partition()

    @property
    def nterms(self) -> int:
        return self._nterms

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
    def groups(self) -> List[FrozenSet[PauliString]]:
        return self._groups

    @property
    def ngroups(self) -> int:
        return self._ngroups

    def partition(self) -> None:
        plists: List[List[PauliString]] = []
        paulis = copy.deepcopy(self._paulis)

        while paulis:
            pauli = paulis.pop()
            added = False
            for (i, plist) in enumerate(plists):
                if all(pauli.can_be_measured_with(p) for p in plist):
                    plists[i].append(pauli)
                    added = True
                    break

            if not added:
                plists.append([pauli])

        self._groups = [frozenset(plist) for plist in plists]
        self._ngroups = len(self._groups)

    def _measure_in(self, circuit: cirq.Circuit) -> List[cirq.Circuit]:
        circuits: List[cirq.Circuit] = []
        base_circuit = copy.deepcopy(circuit)

        for pset in self._groups:
            basis_rotations = set()
            qubits_to_measure = set()
            for pauli in pset:
                basis_rotations.update(pauli._basis_rotations())
                qubits_to_measure.update(pauli._qubits_to_measure())
            circuits.append(
                base_circuit
                + basis_rotations
                + cirq.measure(*sorted(qubits_to_measure))
            )

        return circuits

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

    def expectation(self, circuit: QPROGRAM, executor: Callable[[QPROGRAM], QuantumResult]) -> float:
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

    def _expectation_from_wavefunction(self, wavefunction: np.ndarray) -> float:
        return wavefunction.conj().T @ self.matrix() @ wavefunction

    def _expectation_from_trajectories(self, wavefunctions: List[np.ndarray]) -> float:
        return np.sum([self._expectation_from_wavefunction(wf) for wf in wavefunctions]) / len(wavefunctions)

    def _expectation_from_density_matrix(self, density_matrix: np.ndarray) -> float:
        return np.trace(density_matrix @ self.matrix()).real

    def _expectation_from_measurements(self, measurements: List[MeasurementResult]) -> float:
        expectation = 0.0

        for (group, bitstrings) in zip(self._groups, measurements):
            for pauli in group:
                expectation += pauli._expectation_from_measurements(bitstrings)

        return expectation
