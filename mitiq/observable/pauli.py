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

from collections import Counter
from typing import (
    Any,
    cast,
    Counter as TCounter,
    Dict,
    List,
    Optional,
    Sequence,
    Set,
)

import numpy as np
import numpy.typing as npt
import cirq

from mitiq._typing import QPROGRAM, MeasurementResult
from mitiq.interface import atomic_converter


class PauliString:
    """A `PauliString` is a (tensor) product of single-qubit Pauli gates I, X,
    Y, and Z, with a leading (real or complex) coefficient. `PauliString`s can
    be measured in any `mitiq.QPROGRAM`.
    """

    _string_to_gate_map = {"I": cirq.I, "X": cirq.X, "Y": cirq.Y, "Z": cirq.Z}

    def __init__(
        self,
        spec: str = "",
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

    @property
    def coeff(self) -> complex:
        return self._pauli.coefficient  # type: ignore

    def matrix(
        self,
        qubit_indices_to_include: Optional[List[int]] = None,
    ) -> npt.NDArray[np.complex64]:
        """Returns the (potentially very large) matrix of the PauliString."""
        qubits = (
            [cirq.LineQubit(x) for x in qubit_indices_to_include]
            if qubit_indices_to_include
            else self._pauli.qubits
        )
        return self._pauli.matrix(qubits=qubits)

    def _basis_rotations(self) -> List[cirq.Operation]:
        """Returns the basis rotations needed to measure the PauliString."""
        return [
            op
            for op in self._pauli.to_z_basis_ops()
            if op.gate != cirq.SingleQubitCliffordGate.I
        ]

    def _qubits_to_measure(self) -> Set[cirq.Qid]:
        return set(self._pauli.qubits)

    def measure_in(self, circuit: QPROGRAM) -> QPROGRAM:
        return PauliStringCollection(self).measure_in(circuit)

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

    def support(self) -> Set[int]:
        return {q.x for q in self._pauli.qubits}

    def weight(self) -> int:
        """Returns the weight of the PauliString, i.e., the number of
        non-identity terms in the PauliString.
        """
        return sum(gate != cirq.I for gate in self._pauli.values())

    def _expectation_from_measurements(
        self, measurements: MeasurementResult
    ) -> float:
        return PauliStringCollection(self)._expectation_from_measurements(
            measurements
        )

    def __mul__(self, other: "PauliString") -> "PauliString":
        result = PauliString()
        result._pauli = self._pauli * other._pauli
        return result

    def __eq__(self, other: Any) -> bool:
        return self._pauli == other._pauli

    def __hash__(self) -> int:
        return self._pauli.__hash__()

    def __str__(self) -> str:
        return str(self._pauli)

    def __repr__(self) -> str:
        return repr(self._pauli)


class PauliStringCollection:
    """A collection of PauliStrings that qubit-wise commute and so can be
    measured with a single circuit.
    """

    def __init__(
        self, *paulis: PauliString, check_precondition: bool = True
    ) -> None:
        """Initializes a `PauliStringCollection`.

        Args:
            paulis: PauliStrings to add to the collection.
            check_precondition: If True, raises an error if some of the
                `PauliString`s do not qubit-wise commute.

        Example:
            >>> pcol = PauliStringCollection(
            >>>     PauliString(spec="X"),
            >>>     PauliString(spec="IZ", coeff=-2.2)
            >>> )
            >>> print(pcol)  # X(0) + (-2.2+0j)*Z(1)
            >>> print(pcol.support())  # {0, 1}
            >>>
            >>> # XZ qubit-wise commutes with X(0) and Z(1), so can be added.
            >>> print(pcol.can_add(PauliString(spec="XZ")))  # True.
            >>> pcol.add(PauliString(spec="XZ"))
            >>> print(pcol)  # X(0) + (-2.2+0j)*Z(1) + X(0)*Z(1)
            >>>
            >>> # Z(0) doesn't qubit-wise commute with X(0), so can't be added.
            >>> print(pcol.can_add(PauliString(spec="Z")))  # False.
        """
        self._paulis_by_weight: Dict[int, TCounter[PauliString]] = dict()
        self.add(*paulis, check_precondition=check_precondition)

    def can_add(self, pauli: PauliString) -> bool:
        return all(pauli.can_be_measured_with(p) for p in self.elements)

    def add(
        self, *paulis: PauliString, check_precondition: bool = True
    ) -> None:
        for pauli in paulis:
            if check_precondition and not self.can_add(pauli):
                raise ValueError(
                    f"Cannot add PauliString {pauli} to PauliStringCollection."
                )
            weight = pauli.weight()
            if self._paulis_by_weight.get(weight) is None:
                self._paulis_by_weight[weight] = Counter({pauli})
            else:
                self._paulis_by_weight[weight].update({pauli})

    @property
    def elements(self) -> List[PauliString]:
        return [
            pauli
            for paulis in self._paulis_by_weight.values()
            for pauli in paulis.elements()
        ]

    @property
    def elements_by_weight(self) -> Dict[int, TCounter[PauliString]]:
        return self._paulis_by_weight

    def support(self) -> Set[int]:
        return {cast(cirq.LineQubit, q).x for q in self._qubits_to_measure()}

    def max_weight(self) -> int:
        return max(self._paulis_by_weight.keys(), default=0)

    def min_weight(self) -> int:
        return min(self._paulis_by_weight.keys(), default=0)

    def _qubits_to_measure(self) -> Set[cirq.Qid]:
        qubits: Set[cirq.Qid] = set()
        for pauli in self.elements:
            qubits.update(pauli._pauli.qubits)
        return qubits

    def measure_in(self, circuit: QPROGRAM) -> QPROGRAM:
        return self._measure_in(circuit, self)

    @staticmethod
    @atomic_converter
    def _measure_in(
        circuit: cirq.Circuit, paulis: "PauliStringCollection"
    ) -> cirq.Circuit:
        # Transform circuit to canonical qubit layout.
        qubit_map = dict(
            zip(
                sorted(circuit.all_qubits()),
                cirq.LineQubit.range(len(circuit.all_qubits())),
            )
        )
        circuit = circuit.transform_qubits(lambda q: qubit_map[q])

        if not paulis._qubits_to_measure().issubset(set(circuit.all_qubits())):
            raise ValueError(
                f"Qubit mismatch. The PauliString(s) act on qubits "
                f"{paulis.support()} but the circuit has qubit indices "
                f"{sorted([q for q in circuit.all_qubits()])}."
            )

        basis_rotations = set()
        support = set()
        for pauli in paulis.elements:
            basis_rotations.update(pauli._basis_rotations())
            support.update(pauli._qubits_to_measure())
        measured = circuit + basis_rotations + cirq.measure(*sorted(support))

        # Transform circuit back to original qubits.
        reverse_qubit_map = dict(zip(qubit_map.values(), qubit_map.keys()))
        return measured.transform_qubits(lambda q: reverse_qubit_map[q])

    def _expectation_from_measurements(
        self, measurements: MeasurementResult
    ) -> float:
        total = 0.0
        for pauli in self.elements:
            bitstrings = measurements[sorted(pauli.support())]
            value = (
                np.average([(-1) ** np.sum(bits) for bits in bitstrings])
                if len(bitstrings) > 0
                else 1.0
            )
            total += pauli.coeff * value
        return total

    def __eq__(self, other: Any) -> bool:
        return self._paulis_by_weight == other._paulis_by_weight

    def __len__(self) -> int:
        return len(self.elements)

    def __str__(self) -> str:
        return " + ".join(map(str, self.elements))
