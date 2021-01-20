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

"""Tools for sampling from the noisy decomposition of ideal operations."""

from typing import List, Optional, Tuple, Union
from copy import deepcopy
import numpy as np

import cirq

from mitiq import QPROGRAM
from mitiq.utils import _equal
from mitiq.conversions import convert_to_mitiq, convert_from_mitiq
from mitiq.pec.types import OperationDecomposition


def _sample_sequence(
    ideal_operation: QPROGRAM,
    decompositions: List[OperationDecomposition],
    random_state: Optional[Union[int, np.random.RandomState]] = None,
) -> Tuple[QPROGRAM, int, float]:
    """Samples an implementable sequence from the PEC representation of the
    input ideal operation & returns this sequence as well as its sign and norm.

    For example, if the ideal operation is U with representation U = a A + b B,
    then this function returns A with probability |a| / (|a| + |b|) and B with
    probability |b| / (|a| + |b|). Also returns sign(A) (sign(B)) and |a| + |b|
    if A (B) is sampled.

    Note that the ideal operation can be a sequence of operations (circuit),
    for instance U = V W, as long as a representation is known. Similarly, A
    and B can be sequences of operations (circuits) or just single operations.

    Args:
        ideal_operation: The ideal operation from which an implementable
            sequence is sampled.
        decompositions: A list of decompositions. If no decomposition is
            included for the input `ideal_operation`, a ValueError is raised.
        random_state: Seed for sampling.

    Returns:
        imp_seq: The sampled implementable sequence as QPROGRAM.
        sign: The sign associated to sampled sequence.
        norm: The one-norm of the decomposition coefficients.

    Raises:
        ValueError: If no decomposition is found for the input ideal_operation.
    """
    # Grab the decomposition for the given ideal operation.
    ideal, _ = convert_to_mitiq(ideal_operation)
    operation_decomposition = None
    for decomposition in decompositions:
        if _equal(decomposition.ideal, ideal, require_qubit_equality=True):
            operation_decomposition = decomposition
            break

    if operation_decomposition is None:
        raise ValueError(
            f"Decomposition for ideal operation {ideal_operation} not found "
            f"in provided decompositions."
        )

    # Sample from this decomposition.
    noisy_operation, sign, _ = operation_decomposition.sample(random_state)
    return noisy_operation.ideal_circuit(), sign, operation_decomposition.norm


def _sample_circuit(
    ideal_circuit: QPROGRAM,
    decompositions: List[OperationDecomposition],
    random_state: Optional[Union[int, np.random.RandomState]] = None,
) -> Tuple[QPROGRAM, int, float]:
    """Samples an implementable circuit from the PEC representation of the
    input ideal circuit & returns this circuit as well as its sign and norm.

    This function iterates through each operation in the circuit and samples
    an implementable sequence. The returned sign (norm) is the product of signs
    (norms) sampled for each operation.

    Args:
        ideal_circuit: The ideal circuit from which an implementable circuit
            is sampled.
        decompositions: List of decompositions for every operation in the input
            circuit. If a decomposition cannot be found for an operation in the
            circuit, a ValueError is raised.
        random_state: Seed for sampling.

    Returns:
        imp_circuit: The sampled implementable circuit.
        sign: The sign associated to sampled_circuit.
        norm: The one norm of the decomposition coefficients of the circuit.

    Raises:
        ValueError:
            If a decomposition cannot be found for an operation in the circuit.
    """
    if isinstance(random_state, int):
        random_state = np.random.RandomState(random_state)

    # TODO: Likely to cause issues - conversions may introduce gates which are
    #  not included in `decompositions`.
    ideal, rtype = convert_to_mitiq(ideal_circuit)

    # copy and remove all moments
    sampled_circuit = deepcopy(ideal)[0:0]

    # Iterate over all operations
    sign = 1
    norm = 1.0
    for op in ideal.all_operations():
        imp_seq, loc_sign, loc_norm = _sample_sequence(
            cirq.Circuit(op), decompositions, random_state
        )
        cirq_seq, _ = convert_to_mitiq(imp_seq)
        sign *= loc_sign
        norm *= loc_norm
        sampled_circuit.append(cirq_seq.all_operations())

    return convert_from_mitiq(sampled_circuit, rtype), sign, norm
