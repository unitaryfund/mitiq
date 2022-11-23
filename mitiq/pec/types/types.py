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
from typing import Any, cast, Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
import numpy.typing as npt

import cirq
from cirq.value.linear_dict import _format_coefficient

from mitiq import QPROGRAM
from mitiq.interface import (
    convert_from_mitiq,
    convert_to_mitiq,
    CircuitConversionError,
    UnsupportedCircuitError,
)
from mitiq.utils import _equal


class NoisyOperation:
    """An operation (or sequence of operations) which a noisy quantum computer
    can actually implement.
    """

    def __init__(
        self,
        circuit: QPROGRAM,
        channel_matrix: Optional[npt.NDArray[np.complex64]] = None,
    ) -> None:
        """Initializes a NoisyOperation.

        Args:
            circuit: A short circuit which, when executed on a given noisy
                quantum computer, generates a noisy channel. It typically
                contains a single-gate or a short sequence of gates.
            channel_matrix: Superoperator representation of the noisy channel
                which is generated when executing the input ``circuit`` on the
                noisy quantum computer.

        Raises:
            TypeError: If ``ideal`` is not a ``QPROGRAM``.
        """
        self._native_circuit = circuit

        try:
            ideal_cirq, self._native_type = convert_to_mitiq(circuit)
        except (CircuitConversionError, UnsupportedCircuitError):
            raise TypeError(
                f"Arg `circuit` must be one of {QPROGRAM} but"
                f" was {type(circuit)}."
            )

        self._init_from_cirq(ideal_cirq, channel_matrix)

    @staticmethod
    def from_cirq(
        circuit: cirq.CIRCUIT_LIKE,
        channel_matrix: Optional[npt.NDArray[np.complex64]] = None,
    ) -> "NoisyOperation":
        if isinstance(circuit, cirq.Gate):
            qubits = tuple(cirq.LineQubit.range(circuit.num_qubits()))
            circuit = cirq.Circuit(circuit.on(*qubits))

        elif isinstance(circuit, cirq.Operation):
            circuit = cirq.Circuit(circuit)

        elif isinstance(circuit, cirq.Circuit):
            circuit = deepcopy(circuit)

        else:
            try:
                circuit = cirq.Circuit(circuit)
            except Exception:
                raise ValueError(
                    f"Arg `circuit` must be cirq.CIRCUIT_LIKE "
                    f"but was {type(circuit)}."
                )
        return NoisyOperation(circuit, channel_matrix)

    def _init_from_cirq(
        self,
        circuit: cirq.Circuit,
        channel_matrix: Optional[npt.NDArray[np.complex64]] = None,
    ) -> None:
        """Initializes a noisy operation expressed as a Cirq circuit.

        Args:
            circuit: A circuit which, when executed on a given noisy quantum
                computer, generates a noisy channel.
            channel_matrix: Superoperator representation of the noisy channel
                which is generated when executing the input ``circuit`` on the
                noisy quantum computer.

        Raises:
            ValueError: If the shape of `channel_matrix` does not match the
            shape expected from the size of `circuit`.
        """
        self._circuit = deepcopy(circuit)

        self._qubits = tuple(self._circuit.all_qubits())
        self._num_qubits = len(self._qubits)
        self._dimension = 2**self._num_qubits

        if channel_matrix is None:
            self._channel_matrix = None
            return

        if channel_matrix.shape != (
            self._dimension**2,
            self._dimension**2,
        ):
            raise ValueError(
                f"Arg `channel_matrix` has shape {channel_matrix.shape}"
                " but the expected shape is"
                f" {self._dimension ** 2, self._dimension ** 2}."
            )
        # TODO: Check if channel_matrix is a valid superoperator.
        self._channel_matrix = deepcopy(channel_matrix)

    @staticmethod
    def on_each(
        circuit: cirq.CIRCUIT_LIKE,
        qubits: Sequence[List[cirq.Qid]],
        channel_matrix: Optional[npt.NDArray[np.complex64]] = None,
    ) -> List["NoisyOperation"]:
        """Returns a NoisyOperation(circuit, channel_matrix) on each
        qubit in qubits.

        Args:
            circuit: A gate, operation, sequence of operations, or circuit.
            channel_matrix: Superoperator representation of the noisy channel
                which is generated when executing the input ``circuit`` on the
                noisy quantum computer.
            qubits: The qubits to implement ``circuit`` on.

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
            num_qubits_needed = cirq.num_qubits(circuit)
        except TypeError:
            raise ValueError(
                "Could not deduce number of qubits needed by `circuit`."
            )

        if all(isinstance(q, cirq.Qid) for q in qubits):
            qubits = cast(Sequence[List[cirq.Qid]], [[q] for q in qubits])

        if not all(len(qreg) == num_qubits_needed for qreg in qubits):
            raise ValueError(
                f"Number of qubits in each register should be"
                f" {num_qubits_needed}."
            )

        noisy_ops = []  # type: List[NoisyOperation]
        base_circuit = NoisyOperation.from_cirq(
            circuit,
            channel_matrix,
        )._circuit
        base_qubits = list(base_circuit.all_qubits())

        for new_qubits in qubits:
            try:
                new_qubits = list(iter(new_qubits))
            except TypeError:
                new_qubits = list(new_qubits)

            qubit_map = dict(zip(base_qubits, new_qubits))
            new_circuit = base_circuit.transform_qubits(lambda q: qubit_map[q])

            noisy_ops.append(NoisyOperation(new_circuit, channel_matrix))

        return noisy_ops

    def extend_to(
        self, qubits: Sequence[List[cirq.Qid]]
    ) -> Sequence["NoisyOperation"]:
        return [self] + NoisyOperation.on_each(
            self._circuit,
            qubits,
            self._channel_matrix,
        )

    @staticmethod
    def from_noise_model(
        circuit: cirq.CIRCUIT_LIKE, noise_model: Any
    ) -> "NoisyOperation":
        raise NotImplementedError

    def circuit(self, return_type: Optional[str] = None) -> QPROGRAM:
        """Returns the circuit of the NoisyOperation.

        Args:
            return_type: Type of the circuit to return.
                If not specified, the returned type is the same type as the
                circuit used to initialize the NoisyOperation.
        """
        if not return_type:
            return self._native_circuit
        return convert_from_mitiq(self._circuit, return_type)

    @property
    def qubits(self) -> Tuple[cirq.Qid, ...]:
        return self._qubits

    @property
    def num_qubits(self) -> int:
        return len(self.qubits)

    @property
    def ideal_unitary(self) -> npt.NDArray[np.complex64]:
        return cirq.unitary(self._circuit)

    @property
    def ideal_channel_matrix(self) -> npt.NDArray[np.complex64]:
        raise NotImplementedError

    @property
    def channel_matrix(self) -> npt.NDArray[np.complex64]:
        if self._channel_matrix is None:
            raise ValueError("The channel matrix is unknown.")
        return deepcopy(self._channel_matrix)

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
            qubits = list(iter(cast(Sequence[cirq.Qid], qubits)))
        except TypeError:
            qubits = [cast(cirq.Qid, qubits)]

        if len(qubits) != self._num_qubits:
            raise ValueError(
                f"Expected {self._num_qubits} qubits but received"
                f" {len(qubits)} qubits."
            )

        qubit_map = dict(zip(self._qubits, qubits))
        self._circuit = self._circuit.transform_qubits(lambda q: qubit_map[q])
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
        return NoisyOperation(self._circuit, self._channel_matrix)

    def __add__(self, other: Any) -> "NoisyOperation":
        if not isinstance(other, NoisyOperation):
            raise ValueError(
                f"Arg `other` must be a NoisyOperation but was {type(other)}."
            )

        if self.qubits != other.qubits:
            raise NotImplementedError

        if self._channel_matrix is None or other._channel_matrix is None:
            matrix = None
        else:
            matrix = other._channel_matrix @ self._channel_matrix

        return NoisyOperation(self._circuit + other._circuit, matrix)

    def __str__(self) -> str:
        return self._circuit.__str__()


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

    def add(self, *basis_elements: Sequence["NoisyOperation"]) -> None:
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
                        noisy_op.circuit(return_type="cirq"),  # type: ignore
                        qubits,
                        noisy_op.channel_matrix,
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

    def represent(self, circuit: QPROGRAM) -> None:
        raise NotImplementedError

    def __len__(self) -> int:
        return len(self._basis_elements)


class OperationRepresentation:
    """A decomposition (basis expansion) of an operation or sequence of
    operations in a basis of noisy, implementable operations.
    """

    def __init__(
        self,
        ideal: QPROGRAM,
        basis_expansion: Dict[NoisyOperation, float],
        is_qubit_dependent: bool = True,
    ) -> None:
        """Initializes an OperationRepresentation.

        Args:
            ideal: The ideal operation desired to be implemented.
            basis_expansion: Representation of the ideal operation in a basis
                of `NoisyOperation`s.
            is_qubit_dependent: If True, the representation
                corresponds to the operation on the specific qubits defined in
                `ideal`. If False, the representation is valid for the same
                gate even if acting on different qubits from those specified in
                `ideal`.

        Raises:
            TypeError: If all keys of `basis_expansion` are not instances of
                `NoisyOperation`s.
        """
        self._native_ideal = ideal
        self._ideal, self._native_type = convert_to_mitiq(ideal)
        self.is_qubit_dependent = is_qubit_dependent

        if not all(
            isinstance(op, NoisyOperation) for op in basis_expansion.keys()
        ):
            raise TypeError(
                "All keys of `basis_expansion` must be "
                "of type `NoisyOperation`."
            )

        self._basis_expansion = cirq.LinearDict(basis_expansion)
        self._norm = sum(abs(coeff) for coeff in self.coeffs)
        self._distribution = (
            np.array(
                list(
                    map(
                        abs,  # type: ignore
                        self.coeffs,
                    )
                )
            )
            / self.norm
        )

    @property
    def ideal(self) -> QPROGRAM:
        return self._ideal

    @property
    def basis_expansion(self) -> cirq.LinearDict[NoisyOperation]:
        return self._basis_expansion

    @property
    def noisy_operations(self) -> Tuple[NoisyOperation, ...]:
        return tuple(self._basis_expansion.keys())

    @property
    def coeffs(self) -> Tuple[float, ...]:
        return tuple(cast(List[float], self._basis_expansion.values()))

    @property
    def norm(self) -> float:
        """Returns the L1 norm of the basis expansion coefficients."""
        return self._norm

    def distribution(self) -> npt.NDArray[np.float64]:
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
        return cast(float, self._basis_expansion.get(noisy_op))

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
            rng = random_state  # type: ignore
        else:
            raise TypeError(
                "Arg `random_state` should be of type `np.random.RandomState` "
                f"but was {type(random_state)}."
            )

        noisy_op = rng.choice(
            self.noisy_operations, p=self.distribution()  # type: ignore
        )
        return noisy_op, int(self.sign_of(noisy_op)), self.coeff_of(noisy_op)

    def __str__(self) -> str:
        lhs = str(self._ideal) + " = "
        rhs = ""
        for c, circ in zip(self.coeffs, self.noisy_operations):
            c_str = _format_coefficient(".3f", c)
            if c_str:
                if c_str[0] not in ["+", "-"]:
                    c_str = "+" + c_str
                if len(self._ideal.all_qubits()) == 1:
                    # Print single-qubit circuits horizontally
                    rhs += f"{c_str}*({circ!s})"
                else:
                    # Print multi-qubit circuits vertically
                    rhs += "\n\n" + f"{c_str}\n{circ!s}"
        # Handle special cases as in cirq.value.linear_dict._format_terms()
        if not rhs:
            rhs = f"{0:.3f}"
        # Remove "+" in the first term of a single-qubit representation
        if rhs[0] == "+":
            rhs = rhs[1:]
        # Remove "+" in the first term of a multi-qubit representation
        if rhs[0:3] == "\n\n+":
            rhs = "\n\n" + rhs[3:]
        return lhs + rhs

    def __eq__(self, other: Any) -> bool:
        """Checks if two representations are equivalent. This function return
        True if the representations have the same ideal operation, the same
        coefficients and equivalent NoisyOperation(s) (same gates but not
        necessarily same channel_matrix matrix representations since
        channel_matrix matrices are optional).
        """
        if self._native_type != other._native_type:
            return False

        if not _equal(self._ideal, other._ideal):
            return False

        noisy_ops_a = self.noisy_operations
        noisy_ops_b = other.noisy_operations
        if len(noisy_ops_a) != len(noisy_ops_b):
            return False

        for op_a in noisy_ops_a:
            found = False
            for op_b in noisy_ops_b:
                if _equal(op_a._circuit, op_b._circuit):
                    found = True
                    break
            if not found:
                return False
            if not np.isclose(self.coeff_of(op_a), other.coeff_of(op_b)):
                return False
        return True
