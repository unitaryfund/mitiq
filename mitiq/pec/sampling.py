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

from cirq import Operation, Circuit

from mitiq import QPROGRAM
from mitiq.utils import _equal
from mitiq.conversions import convert_to_mitiq, convert_from_mitiq
from mitiq.pec.types import OperationDecomposition
from mitiq.pec.utils import (
    DecompositionDict,
    get_one_norm,
    get_probabilities,
)


def _sample_sequence(
    ideal_operation: QPROGRAM,
    decompositions: List[OperationDecomposition],
    random_state: Optional[Union[int, np.random.RandomState]] = None,
) -> Tuple[QPROGRAM, int, float]:
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
            Circuit(op), decompositions, random_state
        )
        cirq_seq, _ = convert_to_mitiq(imp_seq)
        sign *= loc_sign
        norm *= loc_norm
        sampled_circuit.append(cirq_seq.all_operations())

    return convert_from_mitiq(sampled_circuit, rtype), sign, norm


def sample_sequence(
    ideal_operation: Operation,
    decomposition_dict: DecompositionDict,
    random_state: Optional[Union[int, np.random.RandomState]] = None,
) -> Tuple[List[Operation], int, float]:
    """Samples an implementable sequence from the PEC decomposition of the
    input ideal operation. Moreover it also returns the "sign" and "norm"
    parameters which are necessary for the Monte Carlo estimation.

    Args:
        ideal_operation: The ideal operation from which an implementable
            sequence is sampled.
        decomposition_dict: The decomposition dictionary from which the
            decomposition of the input ideal_operation can be extracted.
        random_state: Seed for sampling.

    Returns:
        imp_seq: The sampled implementable sequence as list of one
            or more operations.
        sign: The sign associated to sampled sequence.
        norm: The one norm of the decomposition coefficients.
    """
    if random_state is None:
        rng = np.random
    else:
        if isinstance(random_state, int):
            rng = np.random.RandomState(random_state)
        elif isinstance(random_state, np.random.RandomState):
            rng = random_state
        else:
            raise ValueError(
                "Bad type for random_state. Expected int or "
                f"np.random.RandomState but got {type(random_state)}."
            )

    # Extract information from the decomposition dictionary
    probs = get_probabilities(ideal_operation, decomposition_dict)
    one_norm = get_one_norm(ideal_operation, decomposition_dict)

    # Sample an index from the distribution "probs"
    idx = rng.choice(range(len(probs)), p=probs)

    # Get the coefficient and the implementable sequence associated to "idx"
    coeff, imp_seq = decomposition_dict[ideal_operation][idx]

    return imp_seq, int(np.sign(coeff)), one_norm


def sample_circuit(
    ideal_circuit: Circuit,
    decomposition_dict: DecompositionDict,
    random_state: Optional[Union[int, np.random.RandomState]] = None,
) -> Tuple[Circuit, int, float]:
    """Samples an implementable circuit according from the PEC decomposition
    of the input ideal circuit. Moreover it also returns the "sign" and "norm"
    parameters which are necessary for the Monte Carlo estimation.

    Args:
        ideal_circuit: The ideal circuit from which an implementable circuit
            is sampled.
        decomposition_dict: The decomposition dictionary containing the quasi-
            probability representation of the ideal operations (those
            which are part of "ideal_circuit").
        random_state: Seed for sampling.

    Returns:
        imp_circuit: The sampled implementable circuit.
        sign: The sign associated to sampled_circuit.
        norm: The one norm of the decomposition coefficients
            (of the full circuit).
    """
    if isinstance(random_state, int):
        random_state = np.random.RandomState(random_state)

    # copy and remove all moments
    sampled_circuit = deepcopy(ideal_circuit)[0:0]

    # Iterate over all operations
    sign = 1
    norm = 1.0
    for ideal_operation in ideal_circuit.all_operations():
        # Sample an imp. sequence from the decomp. of ideal_operation
        imp_seq, loc_sign, loc_norm = sample_sequence(
            ideal_operation, decomposition_dict, random_state
        )
        sign *= loc_sign
        norm *= loc_norm
        sampled_circuit.append(imp_seq)

    return sampled_circuit, sign, norm
