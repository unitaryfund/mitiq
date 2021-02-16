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
from itertools import product
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np

import cirq

from mitiq import QPROGRAM
from mitiq.conversions import (
    convert_from_mitiq,
    convert_to_mitiq,
    CircuitConversionError,
    UnsupportedCircuitError,
)


class NoisyOperation:
    """An operation (or sequence of operations) which a quantum computer can
     actually implement.
     """

    def __init__(
        self, ideal: QPROGRAM, real: Optional[np.ndarray] = None
    ) -> None:
        """Initializes a NoisyOperation.

        Args:
            ideal: The operation a noiseless quantum computer would implement.
            real: Superoperator representation of the actual operation
                implemented on a noisy quantum computer, if known.

        Raises:
            TypeError: If ideal is not a QPROGRAM.
        """
        self._native_ideal = ideal

        try:
            ideal_cirq, self._native_type = convert_to_mitiq(ideal)
        except (CircuitConversionError, UnsupportedCircuitError):
            raise TypeError(
                f"Arg `ideal` must be one of {QPROGRAM} but was {type(ideal)}."
            )

        self._init_from_cirq(ideal_cirq, real)

    @staticmethod
    def from_cirq(
        ideal: cirq.CIRCUIT_LIKE, real: Optional[np.ndarray] = None
    ) -> "NoisyOperation":
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
        self, ideal: cirq.Circuit, real: Optional[np.ndarray] = None
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
        self._ideal = deepcopy(ideal)

        self._qubits = tuple(self._ideal.all_qubits())
        self._num_qubits = len(self._qubits)
        self._dimension = 2 ** self._num_qubits

        if real is None:
            self._real = None
            return

        if real.shape != (self._dimension ** 2, self._dimension ** 2):
            raise ValueError(
                f"Arg `real` has shape {real.shape} but `ideal` has shape"
                f" {self._dimension ** 2, self._dimension ** 2}."
            )
        # TODO: Check if real is a valid superoperator.
        self._real = deepcopy(real)

    @staticmethod
    def on_each(
        ideal: cirq.CIRCUIT_LIKE,
        qubits: Sequence[List[cirq.Qid]],
        real: Optional[np.ndarray] = None,
    ) -> List["NoisyOperation"]:
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
        base_circuit = NoisyOperation.from_cirq(ideal, real).ideal_circuit()
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
        return [self] + NoisyOperation.on_each(self._ideal, qubits, self._real)

    @staticmethod
    def from_noise_model(
        ideal: cirq.CIRCUIT_LIKE, noise_model
    ) -> "NoisyOperation":
        raise NotImplementedError

    def ideal_circuit(self, return_type: Optional[str] = None) -> cirq.Circuit:
        """Returns the ideal circuit of the NoisyOperation.

        Args:
            return_type: Type of the circuit to return.
                If not specified, the returned type is the same type as the
                circuit used to initialize the NoisyOperation.
        """
        if not return_type:
            return self._native_ideal
        return convert_from_mitiq(self._ideal, return_type)

    @property
    def qubits(self) -> Tuple[cirq.Qid]:
        return self._qubits

    @property
    def num_qubits(self) -> int:
        return len(self.qubits)

    @property
    def ideal_unitary(self) -> np.ndarray:
        return cirq.unitary(self._ideal)

    @property
    def ideal_matrix(self) -> np.ndarray:
        raise NotImplementedError

    @property
    def real_matrix(self) -> np.ndarray:
        if self._real is None:
            raise ValueError("Real matrix is unknown.")
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
        """Returns a copy of the NoisyOperation."""
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

    def __str__(self) -> str:
        return self._ideal.__str__()


class NoisyBasis:
    """A set of noisy operations which a quantum computer can actually
    implement, assumed to form a basis of n-qubit unitary matrices.
    """

    def __init__(self, *basis_elements: NoisyOperation) -> None:
        """Initializes a NoisyBasis.

        Args:
            basis_elements: Sequence of basis elements as `NoisyOperation`s.
        """
        if not all(
            isinstance(element, NoisyOperation) for element in basis_elements
        ):
            raise ValueError(
                "All basis elements must be of type `NoisyOperation`."
            )

        self._basis_elements = set(basis_elements)

    @property
    def elements(self) -> Set[NoisyOperation]:
        return self._basis_elements

    def all_qubits(self) -> Set[cirq.Qid]:
        """Returns the set of qubits that basis elements act on."""
        qubits = set()
        for noisy_op in self._basis_elements:
            qubits.update(set(noisy_op.qubits))
        return qubits

    def add(self, *basis_elements) -> None:
        """Add elements to the NoisyBasis.

        Args:
            basis_elements: Sequence of basis elements as ``NoisyOperation``'s
                to add to the current basis elements.
        """
        for noisy_op in basis_elements:
            if not isinstance(noisy_op, NoisyOperation):
                raise TypeError(
                    "All basis elements must be of type `NoisyOperation`."
                )
            self._basis_elements.add(noisy_op)

    def extend_to(self, qubits: Sequence[List[cirq.Qid]]) -> None:
        """Extends each basis element to act on the provided qubits.

        Args:
            qubits: Additional qubits for each basis element to act on.
        """
        for noisy_op in tuple(self._basis_elements):
            self._basis_elements.update(
                set(
                    NoisyOperation.on_each(
                        noisy_op.ideal_circuit(return_type="cirq"),
                        qubits,
                        noisy_op.real_matrix,
                    )
                )
            )

    def get_sequences(self, length: int) -> List[NoisyOperation]:
        """Returns a list of all implementable NoisyOperation's of the given
        length.

        Example: If the ideal operations of the noisy basis elements are {I, X}
            and length = 2, then this method returns the four NoisyOperations
            whose ideal operations are {II, IX, XI, XX}.

        Args:
            length: Number of NoisyOperation's in each element of the returned
                list.
        """
        sequences = []
        for prod in product(self._basis_elements, repeat=length):
            this_sequence = prod[0]
            for noisy_op in prod[1:]:
                this_sequence += noisy_op
            sequences.append(this_sequence)
        return sequences

    def represent(self, ideal: QPROGRAM):
        raise NotImplementedError

    def __len__(self):
        return len(self._basis_elements)


class OperationRepresentation:
    """A decomposition (basis expansion) of an operation or sequence of
    operations in a basis of noisy, implementable operations.
    """

    def __init__(
        self, ideal: QPROGRAM, basis_expansion: Dict[NoisyOperation, float]
    ) -> None:
        """Initializes an OperationRepresentation.

        Args:
            ideal: The ideal operation desired to be implemented.
            basis_expansion: Representation of the ideal operation in a basis
                of `NoisyOperation`s.

        Raises:
            TypeError: If all keys of `basis_expansion` are not instances of
                `NoisyOperation`s.
        """
        self._native_ideal = ideal
        self._ideal, self._native_type = convert_to_mitiq(ideal)

        if not all(
            isinstance(op, NoisyOperation) for op in basis_expansion.keys()
        ):
            raise TypeError(
                "All keys of `basis_expansion` must be "
                "of type `NoisyOperation`."
            )

        self._basis_expansion = cirq.LinearDict(basis_expansion)
        self._norm = sum(abs(coeff) for coeff in self.coeffs)
        self._distribution = np.array(list(map(abs, self.coeffs))) / self.norm

    @property
    def ideal(self) -> QPROGRAM:
        return self._ideal

    @property
    def basis_expansion(self) -> cirq.LinearDict:
        return self._basis_expansion

    @property
    def noisy_operations(self) -> Tuple[NoisyOperation]:
        return tuple(self._basis_expansion.keys())

    @property
    def coeffs(self) -> Tuple[float]:
        return tuple(self._basis_expansion.values())

    @property
    def norm(self) -> float:
        """Returns the L1 norm of the basis expansion coefficients."""
        return self._norm

    def distribution(self) -> np.ndarray:
        """Returns the Quasi-Probability Representation (QPR) of the
        decomposition. The QPR is the normalized magnitude of each coefficient
        in the basis expansion.
        """
        return self._distribution

    def coeff_of(self, noisy_op: NoisyOperation) -> float:
        """Returns the coefficient of the noisy operation in the basis
        expansion.

        Args:
            noisy_op: NoisyOperation to get the coefficient of.

        Raises:
            ValueError: If noisy_op doesn't appear in the basis expansion.
        """
        if noisy_op not in self.noisy_operations:
            raise ValueError(
                "Arg `noisy_op` does not appear in the basis expansion."
            )
        return self._basis_expansion.get(noisy_op)

    def sign_of(self, noisy_op: NoisyOperation) -> float:
        """Returns the sign of the noisy operation in the basis expansion.

        Args:
            noisy_op: NoisyOperation to get the sign of.

        Raises:
            ValueError: If noisy_op doesn't appear in the basis expansion.
        """
        return np.sign(self.coeff_of(noisy_op))

    def sample(
        self, random_state: Optional[np.random.RandomState] = None
    ) -> Tuple[NoisyOperation, int, float]:
        """Returns a randomly sampled NoisyOperation from the basis expansion.

        Args:
            random_state: Defines the seed for sampling if provided.
        """
        if not random_state:
            rng = np.random
        elif isinstance(random_state, np.random.RandomState):
            rng = random_state
        else:
            raise TypeError(
                "Arg `random_state` should be of type `np.random.RandomState` "
                f"but was {type(random_state)}."
            )

        noisy_op = rng.choice(self.noisy_operations, p=self.distribution())
        return noisy_op, int(self.sign_of(noisy_op)), self.coeff_of(noisy_op)

    def __str__(self):
        # TODO: This works well for one-qubit representations, but doesn't
        #  display nicely in general.
        return str(self._ideal) + " = " + str(self.basis_expansion)
