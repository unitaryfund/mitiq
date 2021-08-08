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
from typing import Any, cast, FrozenSet, List, Optional, Sequence, Set

import numpy as np
import cirq

from mitiq import QPROGRAM
from mitiq.interface import atomic_converter


class PauliString:
    _string_to_gate_map = {"I": cirq.I, "X": cirq.X, "Y": cirq.Y, "Z": cirq.Z}

    def __init__(
        self,
        *,
        spec: str,
        coeff: complex = 1.0,
        support: Optional[Sequence[int]] = None,
    ) -> None:
        """Initialize a PauliString.

        Args:
            spec: String specifier of the PauliString. Should only contain
                characters 'I', 'X', 'Y', and 'Z'.
            coeff: Coefficient of the PauliString.
            support: Qubits the ``spec`` acts on, if provided.

        Examples:
            >>> PauliString(spec="IXY")  # X(1)*Y(2)
            >>> PauliString(spec="ZZ", coeff=-0.5)  # -0.5*Z(0)*Z(1)
            >>> PauliString(spec="XZ", support=(10, 17))  # X(10)*Z(17)
        """
        if not set(spec).issubset(set(self._string_to_gate_map.keys())):
            raise ValueError(
                f"One or more invalid characters in spec {spec}. Valid "
                f"characters are 'I', 'X', 'Y', and 'Z', and the spec should "
                f"not contain any spaces."
            )
        if support is not None:
            if len(support) != len(spec):
                raise ValueError(
                    f"The spec has {len(spec)} Pauli's but the support has "
                    f"{len(support)} qubits. These numbers must be equal."
                )
        else:
            support = range(len(spec))

        self._pauli: cirq.PauliString[Any] = cirq.PauliString(
            coeff,
            (
                self._string_to_gate_map[s].on(cirq.LineQubit(i))
                for (i, s) in zip(support, spec)
            ),
        )

    def matrix(
        self,
        qubit_indices_to_include: Optional[List[int]] = None,
        dtype: type = np.complex128,
    ) -> np.ndarray:
        """Returns the (potentially very large) matrix of the PauliString."""
        qubits = (
            [cirq.LineQubit(x) for x in qubit_indices_to_include]
            if qubit_indices_to_include
            else self._pauli.qubits
        )
        return self._pauli.matrix(qubits=qubits).astype(dtype=dtype)

    def _basis_rotations(self) -> List[cirq.Operation]:
        """Returns the basis rotations needed to measure the PauliString."""
        return [
            op
            for op in self._pauli.to_z_basis_ops()
            if op.gate != cirq.SingleQubitCliffordGate.I
        ]

    def _qubits_to_measure(self) -> Set[cirq.Qid]:
        return set(self._pauli.qubits)

    @staticmethod
    @atomic_converter
    def _measure_in(
        circuit: cirq.Circuit, pauli: "PauliString"
    ) -> cirq.Circuit:
        # Transform circuit to canonical qubit layout.
        qubit_map = dict(
            zip(
                sorted(circuit.all_qubits()),
                cirq.LineQubit.range(len(circuit.all_qubits())),
            )
        )
        circuit = circuit.transform_qubits(lambda q: qubit_map[q])

        if not set(pauli._pauli).issubset(set(circuit.all_qubits())):
            raise ValueError(
                f"Qubit mismatch. The PauliString {pauli._pauli} acts on "
                f"qubits {[q for q in pauli._pauli.qubits]} but the circuit "
                f"has qubit indices "
                f"{sorted([q for q in circuit.all_qubits()])}."
            )

        measured = (
            circuit
            + pauli._basis_rotations()
            + cirq.measure(*sorted(pauli._qubits_to_measure()))
        )

        # Transform circuit back to original qubits.
        reverse_qubit_map = dict(zip(qubit_map.values(), qubit_map.keys()))
        return measured.transform_qubits(lambda q: reverse_qubit_map[q])

    def measure_in(self, circuit: QPROGRAM) -> QPROGRAM:
        return self._measure_in(circuit, self)

    def can_be_measured_with(self, other: "PauliString") -> bool:
        """Returns True if the expectation value of the PauliString can be
        simultaneously estimated with `other` via single-qubit measurements.

        Args:
            other: The PauliString to check simultaneous measurement with.
        """
        overlap = set(self._pauli.qubits).intersection(
            set(other._pauli.qubits)
        )
        for qubit in overlap:
            if cirq.I in (self._pauli.get(qubit), other._pauli.get(qubit)):
                continue
            if self._pauli.get(qubit) != other._pauli.get(qubit):
                return False
        return True

    def weight(self) -> int:
        """Returns the weight of the PauliString, i.e., the number of
        non-identity terms in the PauliString.
        """
        return sum(gate != cirq.I for gate in self._pauli.values())

    def __str__(self) -> str:
        return str(self._pauli)

    def __repr__(self) -> str:
        return repr(self._pauli)


class Observable:
    def __init__(self, *paulis: PauliString) -> None:
        self._paulis = set(paulis)
        self._nterms = len(self._paulis)

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

    def partition(self) -> Set[FrozenSet[PauliString]]:
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

        return set([frozenset(plist) for plist in plists])

    def _measure_in(self, circuit: cirq.Circuit) -> List[cirq.Circuit]:
        circuits: List[cirq.Circuit] = []
        base_circuit = copy.deepcopy(circuit)

        for pset in self.partition():
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
