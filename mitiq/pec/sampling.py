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

"""Tools for sampling from the noisy representations of ideal operations."""

from typing import List, Optional, Tuple, Sequence, Union, cast
from copy import deepcopy
import warnings

import numpy as np

import cirq
from cirq import Circuit

from mitiq import QPROGRAM
from mitiq.interface import convert_to_mitiq, convert_from_mitiq
from mitiq.pec.types import OperationRepresentation
from mitiq.utils import _equal


def sample_sequence(
    ideal_operation: QPROGRAM,
    representations: Sequence[OperationRepresentation],
    random_state: Optional[Union[int, np.random.RandomState]] = None,
    num_samples: int = 1,
) -> Tuple[List[QPROGRAM], List[int], float]:
    """Samples a list of implementable sequences from the quasi-probability
    representation of the input ideal operation.
    Returns the list of sequences, the corresponding list of signs and the
    one-norm of the quasi-probability representation (of the input operation).

    For example, if the ideal operation is U with representation U = a A + b B,
    then this function returns A with probability :math:`|a| / (|a| + |b|)` and
    B with probability :math:`|b| / (|a| + |b|)`. Also returns sign(a)
    (sign(b)) and :math:`|a| + |b|` if A (B) is sampled.

    Note that the ideal operation can be a sequence of operations (circuit),
    for instance U = V W, as long as a representation is known. Similarly, A
    and B can be sequences of operations (circuits) or just single operations.

    Args:
        ideal_operation: The ideal operation from which an implementable
            sequence is sampled.
        representations: A list of representations of ideal operations in a
            noisy basis. If no representation is found for `ideal_operation`,
            a ValueError is raised.
        random_state: Seed for sampling.
        num_samples: The number of samples.

    Returns:
        The tuple (``sequences``, ``signs``, ``norm``) where
        ``sequences`` are the sampled sequences,
        ``signs`` are the signs associated to the sampled ``sequences`` and
        ``norm`` is the one-norm of the quasi-probability distribution.

    Raises:
        ValueError: If no representation is found for `ideal_operation`.
    """
    # Grab the representation for the given ideal operation.
    ideal, _ = convert_to_mitiq(ideal_operation)
    operation_representation = None
    for representation in representations:
        if _equal(
            cast(Circuit, representation.ideal),
            ideal,
            require_qubit_equality=representation.is_qubit_dependent,
        ):
            operation_representation = representation
            break
    if operation_representation is None:
        warnings.warn(
            UserWarning(f"No representation found for \n\n{ideal_operation}.")
        )
        return (
            [ideal_operation] * num_samples,
            [1] * num_samples,
            1.0,
        )

    # Sample from this representation.
    norm = operation_representation.norm
    sequences = []
    signs = []
    for _ in range(num_samples):
        noisy_op, sign, _ = operation_representation.sample(
            random_state  # type: ignore
        )
        sequences.append(noisy_op.circuit())
        signs.append(sign)

    return sequences, signs, norm


def sample_circuit(
    ideal_circuit: QPROGRAM,
    representations: Sequence[OperationRepresentation],
    random_state: Optional[Union[int, np.random.RandomState]] = None,
    num_samples: int = 1,
) -> Tuple[List[QPROGRAM], List[int], float]:
    """Samples a list of implementable circuits from the quasi-probability
    representation of the input ideal circuit.
    Returns the list of circuits, the corresponding list of signs and the
    one-norm of the quasi-probability representation (of the full circuit).

    Args:
        ideal_circuit: The ideal circuit from which an implementable circuit
            is sampled.
        representations: List of representations of every operation in the
            input circuit. If a representation cannot be found for an operation
            in the circuit, a ValueError is raised.
        random_state: Seed for sampling.
        num_samples: The number of samples.

    Returns:
        The tuple (``sampled_circuits``, ``signs``, ``norm``) where
        ``sampled_circuits`` are the sampled implementable circuits,
        ``signs`` are the signs associated to sampled_circuits and
        ``norm`` is the one-norm of the circuit representation.

    Raises:
        ValueError:
            If a representation is not found for an operation in the circuit.
    """
    if isinstance(random_state, int):
        random_state = np.random.RandomState(random_state)

    # TODO: Likely to cause issues - conversions may introduce gates which are
    #  not included in `decompositions`.
    ideal, rtype = convert_to_mitiq(ideal_circuit)

    # copy and remove all moments
    sampled_circuits = [deepcopy(ideal)[0:0] for _ in range(num_samples)]
    sampled_signs = [1 for _ in range(num_samples)]
    norm = 1.0

    for op in ideal.all_operations():
        # Ignore all measurements.
        if cirq.is_measurement(op):
            continue

        sequences, loc_signs, loc_norm = sample_sequence(
            cirq.Circuit(op),
            representations,
            num_samples=num_samples,
            random_state=random_state,
        )

        norm *= loc_norm

        for j in range(num_samples):
            sampled_signs[j] *= loc_signs[j]
            cirq_seq, _ = convert_to_mitiq(sequences[j])
            sampled_circuits[j].append(cirq_seq.all_operations())

    native_circuits = [convert_from_mitiq(c, rtype) for c in sampled_circuits]

    return native_circuits, sampled_signs, norm
