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

from typing import Optional, Sequence

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

        self._pauli = cirq.PauliString(
            coeff,
            (
                self._string_to_gate_map[s].on(cirq.LineQubit(i))
                for (i, s) in zip(support, spec)
            ),
        )

    def matrix(self) -> np.ndarray:
        """Returns the (potentially very large) matrix of the PauliString."""
        return self._pauli.matrix()

    def measure_in(self, circuit: QPROGRAM) -> QPROGRAM:
        @atomic_converter
        def _measure_in(circuit: cirq.Circuit, pauli: cirq.PauliString):
            # Transform circuit to canonical qubit layout.
            qubit_map = dict(
                zip(
                    sorted(circuit.all_qubits()),
                    cirq.LineQubit.range(len(circuit.all_qubits())),
                )
            )
            circuit = circuit.transform_qubits(lambda q: qubit_map[q])

            # Measure the Paulis.
            if not set(pauli).issubset(set(circuit.all_qubits())):
                raise ValueError(
                    f"Qubit mismatch. The PauliString {self} acts on qubits "
                    f"{[q.x for q in pauli.qubits]} but the circuit has qubit "
                    f"indices {sorted([q.x for q in circuit.all_qubits()])}."
                )
            measured = (
                circuit + pauli.to_z_basis_ops() + cirq.measure(*pauli.qubits)
            )

            # Transform circuit back to original qubits.
            reverse_qubit_map = dict(zip(qubit_map.values(), qubit_map.keys()))
            return measured.transform_qubits(lambda q: reverse_qubit_map[q])

        return _measure_in(circuit, self._pauli)

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
