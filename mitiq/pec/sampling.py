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

from typing import Tuple, List
import numpy as np

from cirq import Operation

from mitiq.pec.utils import (
    DecoType,
    get_one_norm,
    get_probabilities,
)


def sample_sequence(
    ideal_operation: Operation, deco_dict: DecoType,
) -> Tuple[List[Operation], int, float]:
    """Samples an implementable sequence from the PEC decomposition of the
    input ideal operation. Moreover it also returns the "sign" and "norm"
    parameters which are necessary for the Monte Carlo estimation.

    Args:
        ideal_operation = The ideal operation from which an implementable
            sequence is sampled.
        deco_dict = The decomposition dictionary from which the decomposition
            of the input ideal_operation can be extracted.

    Returns:
        imp_seq: The sampled implementable sequence as list of one
            or more operations.
        sign: The sign associated to sampled sequence.
        norm: The one norm of the decomposition coefficients.
    """
    # Extract information from the decomposition dictionary
    probs = get_probabilities(ideal_operation, deco_dict)
    one_norm = get_one_norm(ideal_operation, deco_dict)

    # Sample an index from the distribution "probs"
    idx = np.random.choice(list(range(len(probs))), p=probs)

    # Get the coefficient and the implementanble sequence associated to "idx"
    coeff, imp_seq = deco_dict[ideal_operation][idx]

    return imp_seq, np.sign(coeff), one_norm
