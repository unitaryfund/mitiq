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

# TODO: Functions which don't fit in pec.py and sampling.py are placed here.
# Some of them could be moved in future new sub-modules of PEC
# (e.g. decomposition, tomo, etc.)

"""Utilities related to probabilistic error cancellation."""

import numpy as np
from typing import Tuple, List, Dict
from cirq import Operation, I, X, Y, Z, CNOT, LineQubit

# Type definition for a decomposition dictionary.
# Keys are ideal operations.
# Values describe the associated decompositions.
DecoType = Dict[Operation, List[Tuple[float, List[Operation]]]]


def _simple_pauli_deco_dict(
    base_noise: float, simplify_paulis: bool = False
) -> DecoType:
    """Returns a simple hard-coded decomposition
    dictionary to be used for testing and protoptyping.

    The decomposition is compatible with one-qubit or
    two-qubit circuits involving only Pauli and CNOT gates.

    The keys of the output dictionary are Pauli and CNOT operations.

    The decomposition assumes that Pauli and CNOT operations,
    followed by local depolarizing noise, are implementable.

    Args:
        base_noise: The depolarizing noise level.
        simplify_paulis: If True, products of Paulis are simplified to a
            single Pauli. If False, Pauli sequences are not simplified.

    Returns:
        deco_dict: The decomposition dictionary.

    """
    # Initialize two qubits
    qreg = LineQubit.range(2)

    # Single-qubit Pauli operations
    i0 = I.on(qreg[0])
    x0 = X.on(qreg[0])
    y0 = Y.on(qreg[0])
    z0 = Z.on(qreg[0])
    i1 = I.on(qreg[1])
    x1 = X.on(qreg[1])
    y1 = Y.on(qreg[1])
    z1 = Z.on(qreg[1])
    single_paulis = [x0, y0, z0, x1, y1, z1]

    # Single-qubit decomposition coefficients
    epsilon = base_noise * 4 / 3
    c_neg = -(1 / 4) * epsilon / (1 - epsilon)
    c_pos = 1 - 3 * c_neg
    assert np.isclose(c_pos + 3 * c_neg, 1.0)

    # Single-qubit decomposition dictionary
    deco_dict = {}
    if simplify_paulis:
        # Hard-coded simplified gates
        deco_dict = {
            x0: [(c_pos, [x0]), (c_neg, [i0]), (c_neg, [z0]), (c_neg, [y0])],
            y0: [(c_pos, [y0]), (c_neg, [z0]), (c_neg, [i0]), (c_neg, [x0])],
            z0: [(c_pos, [z0]), (c_neg, [y0]), (c_neg, [x0]), (c_neg, [i0])],
            x1: [(c_pos, [x1]), (c_neg, [i1]), (c_neg, [z1]), (c_neg, [y1])],
            y1: [(c_pos, [y1]), (c_neg, [z1]), (c_neg, [i1]), (c_neg, [x1])],
            z1: [(c_pos, [z1]), (c_neg, [y1]), (c_neg, [x1]), (c_neg, [i1])],
        }
    else:
        for local_paulis in [[x0, y0, z0], [x1, y1, z1]]:
            for key in local_paulis:
                key_deco_pos = [(c_pos, [key])]
                key_deco_neg = [(c_neg, [key, op]) for op in local_paulis]
                deco_dict[key] = key_deco_pos + key_deco_neg  # type: ignore

    # Two-qubit Paulis
    xx = [x0, x1]
    xy = [x0, y1]
    xz = [x0, z1]
    yx = [y0, x1]
    yy = [y0, y1]
    yz = [y0, z1]
    zx = [z0, x1]
    zy = [z0, y1]
    zz = [z0, z1]
    double_paulis = [xx, xy, xz, yx, yy, yz, zx, zy, zz]

    # Two-qubit decomposition coefficients (assuming local noise)
    c_pos_pos = c_pos * c_pos
    c_pos_neg = c_neg * c_pos
    c_neg_neg = c_neg * c_neg
    assert np.isclose(c_pos_pos + 6 * c_pos_neg + 9 * c_neg_neg, 1.0)

    cnot = CNOT.on(qreg[0], qreg[1])
    cnot_decomposition = [(c_pos_pos, [cnot])]
    for p in single_paulis:
        cnot_decomposition.append((c_pos_neg, [cnot] + [p]))
    for pp in double_paulis:
        cnot_decomposition.append((c_neg_neg, [cnot] + pp))  # type: ignore

    deco_dict[cnot] = cnot_decomposition  # type: ignore

    return deco_dict  # type: ignore


def get_coefficients(
    ideal_operation: Operation, deco_dict: DecoType
) -> List[float]:
    """Extracts, from the input decomposition dictionary, the decomposition
    coefficients associated to the input ideal_operation.

    Args:
        ideal_operation: The input ideal operation.
        deco_dict: The input decomposition dictionary.

    Returns:
        The decomposition coefficients of the input operation.
    """
    op_decomp = deco_dict[ideal_operation]

    return [coeff_and_seq[0] for coeff_and_seq in op_decomp]


def get_imp_sequences(
    ideal_operation: Operation, deco_dict: DecoType
) -> List[List[Operation]]:
    """Extracts, from the input decomposition dictionary, the list of
    implementable sequences associated to the input ideal_operation.

    Args:
        ideal_operation: The input ideal operation.
        deco_dict: The input decomposition dictionary.

    Returns:
        The list of implementable sequences.
    """
    op_decomp = deco_dict[ideal_operation]

    return [coeff_and_seq[1] for coeff_and_seq in op_decomp]


def get_one_norm(ideal_operation: Operation, deco_dict: DecoType) -> float:
    """Extracts, from the input decomposition dictionary, the one-norm
    (i.e. the sum of absolute values) of the the decomposition coefficients
    associated to the input ideal_operation.

    Args:
        ideal_operation: The input ideal operation.
        deco_dict: The input decomposition dictionary.

    Returns:
        The one-norm of the decomposition coefficients.
    """
    coeffs = get_coefficients(ideal_operation, deco_dict)
    return np.linalg.norm(coeffs, ord=1)


def get_probabilities(
    ideal_operation: Operation, deco_dict: DecoType
) -> List[float]:
    """Evaluates, from the input decomposition dictionary, the normalized
    probability distribution associated to the input ideal_operation.

    Sampling implementable sequences with this distribution (taking
    into account the corresponding "sign") approximates the exact
    decomposition of the input ideal_operation.

    Args:
        ideal_operation: The input ideal operation.
        deco_dict: The input decomposition dictionary.

    Returns:
        The probability distribution suitable for Monte Carlo sampling.
    """
    coeffs = get_coefficients(ideal_operation, deco_dict)
    return list(np.abs(coeffs) / np.linalg.norm(coeffs, ord=1))
