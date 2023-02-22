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
from typing import Any, cast, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import numpy.typing as npt

import cirq
from cirq.value.linear_dict import _format_coefficient

from mitiq import QPROGRAM
from mitiq.interface import (
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
        self._native_circuit = deepcopy(circuit)

        try:
            cirq_circuit, native_type = convert_to_mitiq(circuit)
        except (CircuitConversionError, UnsupportedCircuitError):
            raise TypeError(
                "Failed to convert to an internal Mitiq representation"
                f"the input circuit:\n{type(circuit)}\n"
            )

        self._circuit = cirq_circuit
        self._native_type = native_type

        dimension = 2**self.num_qubits

        if channel_matrix is None:
            self._channel_matrix = None

        elif channel_matrix.shape != (
            dimension**2,
            dimension**2,
        ):
            raise ValueError(
                f"Arg `channel_matrix` has shape {channel_matrix.shape}"
                " but the expected shape is"
                f" {dimension ** 2, dimension ** 2}."
            )
        self._channel_matrix = deepcopy(channel_matrix)

    @property
    def circuit(self) -> cirq.Circuit:
        """Returns the circuit of the NoisyOperation as a Cirq circuit."""
        return self._circuit

    @property
    def native_circuit(self) -> QPROGRAM:
        """Returns the circuit used to initialize the NoisyOperation."""
        return self._native_circuit

    @property
    def qubits(self) -> Tuple[cirq.Qid, ...]:
        return tuple(self._circuit.all_qubits())

    @property
    def num_qubits(self) -> int:
        return len(self.qubits)

    @property
    def channel_matrix(self) -> npt.NDArray[np.complex64]:
        if self._channel_matrix is None:
            raise ValueError("The channel matrix is unknown.")
        return deepcopy(self._channel_matrix)

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
        return self._native_circuit.__str__()


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
