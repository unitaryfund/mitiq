# Copyright (C) Unitary Fund
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

"""Functions for multivariate richardson extrapolation as defined in
:cite:`Russo_2024_LRE`.
"""

import warnings
from collections import Counter
from itertools import combinations_with_replacement
from typing import Any, Dict, List, Optional

import numpy as np
from cirq import Circuit
from numpy.typing import NDArray

from mitiq.lre.multivariate_scaling.layerwise_folding import (
    _get_scale_factor_vectors,
)


def _full_monomial_basis_term_exponents(
    num_layers: int, degree: int
) -> List[Dict[int, int]]:
    """Exponents of monomial terms required to create the sample matrix."""
    variables = [i for i in range(1, num_layers + 1)]
    variable_combinations = []
    for d in range(degree, -1, -1):
        for var_tuple in combinations_with_replacement(variables, d):
            variable_combinations.append(var_tuple)

    variable_exp_counter = [
        dict(Counter(term)) for term in variable_combinations
    ]
    # make sure all variable numbers are used as keys
    # if something is not in the created dictionary, add the key and exponent
    # combination. Exponent is 0 in this case.
    for combo_key in variable_exp_counter:
        for j in variables:
            if j not in combo_key:
                combo_key[j] = 0
    # return a dictionary with variable number as the key with exponent values
    # first term is the 0 degree term's exponent
    return variable_exp_counter[::-1]


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

    # replace first row and column of the sample matrix by 1s
    sample_matrix[:, 0] = 1.0
    sample_matrix[0, :] = 1.0
    # skip first element of the tuple due to above replacements
    variable_exp_wout_0_degree = variable_exp[1:]

    # sort dict
    variable_exp_wout_0_degree_list = []
    for i in variable_exp_wout_0_degree:
        variable_exp_wout_0_degree_list.append(dict(sorted(i.items())))
    variable_exp_no_0_degree = variable_exp_wout_0_degree_list

    # create a list of dict values (just the exponents)
    variable_exp_list = []
    for i in variable_exp_no_0_degree:
        val_i = list(i.values())
        variable_exp_list.append(val_i)

    for rows, i in enumerate(scale_factor_vectors[1:], start=1):  # type: ignore[assignment]
        for cols, j in enumerate(variable_exp_list, start=1):
            evaluated_terms = []
            for base, exp in zip(list(i), j):
                # raise scale factor value by the exponent dict value
                evaluated_terms.append(base**exp)
            # multiply both elements in the list to create an evaluated
            # monomial term
            sample_matrix[rows, cols] = np.prod(evaluated_terms)

    return sample_matrix


def linear_combination_coefficients(
    input_circuit: Circuit,
    degree: int,
    fold_multiplier: int,
    num_chunks: Optional[int] = None,
) -> List[np.float64]:
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
        List of the evaluated monomial basis terms using the scale factor
            vectors.
    """
    num_layers = len(
        _get_scale_factor_vectors(
            input_circuit, degree, fold_multiplier, num_chunks
        )
    )
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
        det = np.exp(logdet)  # pragma: no cover

    if np.isinf(det):
        raise ValueError(  # pragma: no cover
            "Determinant of sample matrix cannot be calculated as "
            + "the matrix is too large. Consider chunking your"
            + " input circuit. "
        )
    assert det != 0.0

    coeff_list = []
    for i in range(num_layers):
        sample_matrix_copy = input_sample_matrix.copy()
        sample_matrix_copy[i] = np.array([[1] + [0] * (num_layers - 1)])
        coeff_list.append(np.linalg.det(sample_matrix_copy) / det)

    return coeff_list
