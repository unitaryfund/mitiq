# Copyright (C) Unitary Fund
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

"""Functions for multivariate richardson extrapolation as defined in
:cite:`Russo_2024_LRE`.
"""

import warnings
from itertools import product
from typing import Any, Optional

import numpy as np
from cirq import Circuit
from numpy.typing import NDArray

from mitiq.lre.multivariate_scaling.layerwise_folding import (
    _get_scale_factor_vectors,
)


def _full_monomial_basis_term_exponents(
    num_layers: int, degree: int
) -> list[tuple[int, ...]]:
    """Exponents of monomial terms required to create the sample matrix."""
    exponents = {
        exps
        for exps in product(range(degree + 1), repeat=num_layers)
        if sum(exps) <= degree
    }

    return sorted(exponents, key=lambda term: (sum(term), term[::-1]))


def sample_matrix(
    input_circuit: Circuit,
    degree: int,
    fold_multiplier: int,
    num_chunks: Optional[int] = None,
) -> NDArray[Any]:
    r"""
    Defines the sample matrix required for multivariate extrapolation as
    defined in :cite:`Russo_2024_LRE`.

    Args:
        input_circuit: Circuit to be scaled.
        degree: Degree of the multivariate polynomial.
        fold_multiplier: Scaling gap required by unitary folding.
        num_chunks: Number of desired approximately equal chunks. When the
            number of chunks is the same as the layers in the input circuit,
            the input circuit is unchanged.

    Returns:
        Matrix of the evaluated monomial basis terms from the scale factor
            vectors.

    Raises:
        ValueError:
            When the degree for the multinomial is not greater than or
                equal to 1; when the fold multiplier to scale the circuit is
                greater than/equal to 1; when the number of chunks for a
                large circuit is 0 or when the number of chunks in a circuit is
                greater than the number of layers in the input circuit.

    """
    if degree < 1:
        raise ValueError(
            "Multinomial degree must be greater than or equal to 1."
        )
    if fold_multiplier < 1:
        raise ValueError("Fold multiplier must be greater than or equal to 1.")

    scale_factor_vectors = _get_scale_factor_vectors(
        input_circuit, degree, fold_multiplier, num_chunks
    )
    num_layers = len(scale_factor_vectors[0])

    # Evaluate the monomial terms using the values in the scale factor vectors
    # and insert in the sample matrix
    # each row is specific to each scale factor vector
    # each column is a term in the monomial basis
    variable_exp = _full_monomial_basis_term_exponents(num_layers, degree)
    sample_matrix = np.empty((len(variable_exp), len(variable_exp)))

    for i, scale_factors in enumerate(scale_factor_vectors):
        for j, exponent in enumerate(variable_exp):
            evaluated_terms = []
            for base, exp in zip(scale_factors, exponent):
                # raise scale factor value by the exponent dict value
                evaluated_terms.append(base**exp)
            # multiply both elements in the list to create an evaluated
            # monomial term
            sample_matrix[i, j] = np.prod(evaluated_terms)

    return sample_matrix


def linear_combination_coefficients(
    input_circuit: Circuit,
    degree: int,
    fold_multiplier: int,
    num_chunks: Optional[int] = None,
) -> list[float]:
    r"""
    Defines the function to find the linear combination coefficients from the
    sample matrix as required for multivariate extrapolation (defined in
    :cite:`Russo_2024_LRE`).

    Args:
        input_circuit: Circuit to be scaled.
        degree: Degree of the multivariate polynomial.
        fold_multiplier: Scaling gap required by unitary folding.
        num_chunks: Number of desired approximately equal chunks. When the
            number of chunks is the same as the layers in the input circuit,
            the input circuit is unchanged.

    Returns:
        List of the evaluated monomial basis terms using the scale factor
            vectors.
    """
    input_sample_matrix = sample_matrix(
        input_circuit, degree, fold_multiplier, num_chunks
    )
    try:
        det = np.linalg.det(input_sample_matrix)
    except RuntimeWarning:  # pragma: no cover
        # taken from https://stackoverflow.com/a/19317237
        warnings.warn(  # pragma: no cover
            "To account for overflow error, required determinant of "
            + "large sample matrix is calculated through "
            + "`np.linalg.slogdet`."
        )
        sign, logdet = np.linalg.slogdet(  # pragma: no cover
            input_sample_matrix
        )
        det = sign * np.exp(logdet)  # pragma: no cover

    if np.isinf(det):
        raise ValueError(  # pragma: no cover
            "Determinant of sample matrix cannot be calculated as "
            + "the matrix is too large. Consider chunking your"
            + " input circuit. "
        )
    assert det != 0.0

    coeff_list = []
    mat_row, mat_cols = input_sample_matrix.shape
    assert mat_row == mat_cols
    # replace a row of the sample matrix with [1, 0, 0, .., 0]
    repl_row = np.array([[1] + [0] * (mat_cols - 1)])
    for i in range(mat_row):
        if i == 0:  # first row
            new_mat = np.concatenate(
                (repl_row, input_sample_matrix[1:]), axis=0
            )
        elif i == mat_row - 1:  # last row
            new_mat = np.concatenate(
                (input_sample_matrix[:i], repl_row), axis=0
            )
        else:
            frst_sl = np.concatenate(
                (input_sample_matrix[:i], repl_row), axis=0
            )
            sec_sl = input_sample_matrix[i + 1 :]
            new_mat = np.concatenate((frst_sl, sec_sl), axis=0)

        coeff_list.append(np.linalg.det(new_mat) / det)

    return coeff_list
