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

"""Types used in probabilistic error cancellation."""
from copy import deepcopy
from typing import Any, List, Sequence, Tuple, Union

import numpy as np

import cirq

from mitiq import QPROGRAM
from mitiq.conversions import convert_from_mitiq, convert_to_mitiq


class NoisyOperation:
    """An operation (or sequence of operations) which a quantum computer can
     actually implement.
     """
    def __init__(self, ideal: QPROGRAM, real: np.ndarray) -> None:
        self._native_ideal = ideal
        ideal_cirq, self._native_type = convert_to_mitiq(ideal)
        self._init_from_cirq(ideal_cirq, real)

    @staticmethod
    def from_cirq(
            ideal: cirq.CIRCUIT_LIKE, real: np.ndarray
    ) -> 'NoisyOperation':
        if isinstance(ideal, cirq.Gate):
            qubits = tuple(cirq.LineQubit.range(ideal.num_qubits()))
            ideal = cirq.Circuit(ideal.on(*qubits))

        elif isinstance(ideal, cirq.Operation):
            ideal = cirq.Circuit(ideal)

        elif isinstance(ideal, cirq.Circuit):
            ideal = deepcopy(ideal)

        else:
            try:
                ideal = cirq.Circuit(ideal)
            except Exception:
                raise ValueError(
                    f"Arg `ideal` must be cirq.CIRCUIT_LIKE "
                    f"but was {type(ideal)}."
                )
        return NoisyOperation(ideal, real)

    def _init_from_cirq(
            self, ideal: cirq.Circuit, real: np.ndarray
    ) -> None:
        """Initializes a noisy operation expressed as a Cirq circuit.

        Args:
            ideal: An ideal (noiseless) circuit.
            real: Superoperator of the ideal circuit performed when
                implemented on a quantum processor.

        Raises:
            ValueError:
                * If the shape of `real` does not match the expected shape
                    from `ideal`.
        """
        self._qubits = tuple(ideal.all_qubits())
        self._num_qubits = len(self._qubits)
        self._dimension = 2 ** self._num_qubits

        if real.shape != (self._dimension ** 2, self._dimension ** 2):
            raise ValueError(
                f"Arg `real` has shape {real.shape} but `ideal` has shape"
                f" {self._dimension ** 2, self._dimension ** 2}."
            )
        # TODO: Check if real is a valid superoperator.

        self._ideal = deepcopy(ideal)
        self._real = deepcopy(real)

    @staticmethod
    def on_each(
        ideal: cirq.CIRCUIT_LIKE,
        real: np.ndarray,
        qubits: Sequence[List[cirq.Qid]],
    ) -> List['NoisyOperation']:
        """Returns a NoisyOperation(ideal, real) on each qubit in qubits.

        Args:
            ideal: An ideal (noiseless) gate, operation, sequence of
                operations, or circuit.
            real: Superoperator of the ideal operation performed when
                implemented on a quantum processor.
            qubits: The qubits to implement `ideal` on.

        Raises:
            TypeError:
                * If `qubits` is not iterable.
                * If `qubits` is not an iterable of cirq.Qid's or
                  a sequence of lists of cirq.Qid's of the same length.
        """
        try:
            qubits = list(iter(qubits))
        except TypeError:
            raise TypeError("Argument `qubits` must be iterable.")

        try:
            num_qubits_needed = cirq.num_qubits(ideal)
        except TypeError:
            raise ValueError(
                "Could not deduce number of qubits needed by `ideal`."
            )

        if all(isinstance(q, cirq.Qid) for q in qubits):
            qubits = [[q] for q in qubits]

        if not all(len(qreg) == num_qubits_needed for qreg in qubits):
            raise ValueError(
                f"Number of qubits in each register should be"
                f" {num_qubits_needed}."
            )

        noisy_ops = []  # type: List[NoisyOperation]
        base_circuit = NoisyOperation(ideal, real).ideal_circuit
        base_qubits = list(base_circuit.all_qubits())

        for new_qubits in qubits:
            try:
                new_qubits = list(iter(new_qubits))
            except TypeError:
                new_qubits = list(new_qubits)

            qubit_map = dict(zip(base_qubits, new_qubits))
            new_circuit = base_circuit.transform_qubits(lambda q: qubit_map[q])

            noisy_ops.append(NoisyOperation(new_circuit, real))

        return noisy_ops

    def extend_to(
        self, qubits: Sequence[List[cirq.Qid]]
    ) -> Sequence["NoisyOperation"]:
        return [self] + NoisyOperation.on_each(self._ideal, self._real, qubits)

    @staticmethod
    def from_noise_model(
        ideal: cirq.CIRCUIT_LIKE, noise_model
    ) -> "NoisyOperation":
        raise NotImplementedError

    @property
    def ideal_circuit(self) -> cirq.Circuit:
        return self._ideal

    @property
    def qubits(self) -> Tuple[cirq.Qid]:
        return self._qubits

    @property
    def num_qubits(self) -> int:
        return len(self.qubits)

    @property
    def ideal_matrix(self) -> np.ndarray:
        return cirq.unitary(self._ideal)

    @property
    def real_matrix(self) -> np.ndarray:
        return deepcopy(self._real)

    def transform_qubits(
        self, qubits: Union[cirq.Qid, Sequence[cirq.Qid]]
    ) -> None:
        """Changes the qubit(s) that the noisy operation acts on.

        Args:
            qubits: Qubit(s) that the noisy operation will act on.

        Raises:
            ValueError: If the number of qubits does not match that
                of the noisy operation.
        """
        try:
            qubits = list(iter(qubits))
        except TypeError:
            qubits = [qubits]

        if len(qubits) != self._num_qubits:
            raise ValueError(
                f"Expected {self._num_qubits} qubits but received"
                f" {len(qubits)} qubits."
            )

        qubit_map = dict(zip(self._qubits, qubits))
        self._ideal = self._ideal.transform_qubits(lambda q: qubit_map[q])
        self._qubits = tuple(qubits)

    def with_qubits(self, qubits: Sequence[cirq.Qid]) -> "NoisyOperation":
        """Returns the noisy operation acting on the input qubits.

        Args:
            qubits: Qubits that the returned noisy operation will act on.

        Raises:
            ValueError: If the number of qubits does not match that
                of the noisy operation.
        """
        copy = self.copy()
        copy.transform_qubits(qubits)
        return copy

    def copy(self) -> "NoisyOperation":
        """Returns a copy of the noisy operation."""
        return NoisyOperation(self._ideal, self._real)

    def __add__(self, other: Any) -> "NoisyOperation":
        if not isinstance(other, NoisyOperation):
            raise ValueError(
                f"Arg `other` must be a NoisyOperation but was {type(other)}."
            )

        if self.qubits != other.qubits:
            raise NotImplementedError

        return NoisyOperation(
            self._ideal + other._ideal, self._real @ other._real
        )

    def __repr__(self) -> str:
        return self._ideal.__repr__() + "\n:\n" + self._real.__repr__()

    def __str__(self) -> str:
        return self._ideal.__str__()
