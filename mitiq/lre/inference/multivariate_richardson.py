# Copyright (C) Unitary Fund
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

"""Functions for multivariate richardson extrapolation as defined in
:cite:`Russo_2024_LRE`.
"""

from collections import Counter
from itertools import chain, combinations_with_replacement
from typing import Any, List, Optional

import numpy as np
from cirq import Circuit
from numpy.typing import NDArray

from mitiq.lre.multivariate_scaling.layerwise_folding import (
    _get_scale_factor_vectors,
)


def _get_variables(num_layers: int) -> List[str]:
    r"""Find the variables required for a certain number of layers in the
    input circuit and degree.

    Args:
        num_layers: Number of layers in the input circuit.

    Returns:
        Variables required to create the monomial basis.
    """
    return [f"λ_{i}" for i in range(1, num_layers + 1)]


def _create_variable_combinations(num_layers: int, degree: int) -> List[Any]:
    """Find the variable combinations required to create the monomial terms.

    Args:
        num_layers: Number of layers in the input circuit.
        degree: Degree of the multivariate polynomial.

    Returns:
        Variable combinations required for the monomial basis.
    """
    variables = _get_variables(num_layers)
    variable_combinations = []
    for i in range(degree, -1, -1):
        # Generate combinations for the current degree.
        # Ranges from max degree, max degree -1, ..., 0.
        combos = list(combinations_with_replacement(variables, i))
        variable_combinations.append(combos)

    # Return a flattened list.
    return list(chain(*variable_combinations))


def full_monomial_basis(num_layers: int, degree: int) -> List[str]:
    r"""Find the monomial basis terms for a number of layers in the input
    and max degree d.

    Number of layers in the input circuit dictate the number of variables
    utilized in the combinations used for the polynomial.

    Degree of the polynomial dictates the max and min degree of the monomial
    terms.

    Args:
        num_layers: Number of layers in the input circuit.
        degree: Degree of the multivariate polynomial.

    Returns:
        Monomial basis terms required for multivariate
            extrapolation up to max degree
    """
    variable_combinations = _create_variable_combinations(num_layers, degree)

    monomial_basis = []
    for combo in variable_combinations:
        monomial_parts = []
        counts = Counter(combo)
        # Ensure variables are processed in lexicographical order
        for var in sorted(counts.keys()):
            count = counts[var]
            if count > 1:
                monomial_parts.append(f"{var}**{count}")
            else:
                monomial_parts.append(var)
        monomial = "*".join(monomial_parts)
        # Handle the case where degree is 0
        # the tuple in variable_combinations is empty
        if not monomial:
            monomial = "1"
        monomial_basis.append(monomial)
    # "1" should be the first monomial (degree = 0).
    # max degree should be the last term
    return monomial_basis[::-1]


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

    monomial_terms = full_monomial_basis(num_layers, degree)
    if len(monomial_terms) != len(scale_factor_vectors):  # pragma: no cover
        # Temporarily ignore this block from the coverage report because
        # a unit test for this is not that obvious.
        raise ValueError("Sample matrix will not be a square matrix.")
    sample_matrix = np.zeros((len(monomial_terms), len(monomial_terms)))

    # Evaluate the monomial terms using the values in the scale factor vectors
    # and insert in the sample matrix
    # each row is specific to each scale factor vector
    # each column is a term in the monomial basis
    for i, point in enumerate(scale_factor_vectors):
        for j, monomial in enumerate(monomial_terms):
            var_mapping = {f"λ_{k+1}": point[k] for k in range(num_layers)}
            sample_matrix[i, j] = eval(monomial, {}, var_mapping)
    return sample_matrix


def linear_combination_coefficients(
    input_circuit: Circuit,
    degree: int,
    fold_multiplier: int,
    num_chunks: Optional[int] = None,
) -> List[int]:
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
        Matrix of the evaluated monomial basis terms using the scale factor
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

    coeff_list = []
    for i in range(num_layers):
        sample_matrix_copy = input_sample_matrix.copy()
        sample_matrix_copy[i] = np.array([[1] + [0] * (num_layers - 1)])
        coeff_list.append(
            np.linalg.det(sample_matrix_copy)
            / np.linalg.det(input_sample_matrix)
        )

    return coeff_list
