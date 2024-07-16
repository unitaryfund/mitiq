# Copyright (C) Unitary Fund
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

"""Functions for the multivariate richardson extrapolation as defined in
:cite:`Russo_2024_LRE`.
"""

from collections import Counter
from itertools import chain, combinations_with_replacement
from typing import List, Optional

import numpy as np
from cirq import Circuit

from mitiq.lre.multivariate_scaling.layerwise_folding import (
    _get_num_layers_without_measurements,
    _get_scale_factor_vectors,
)


def _get_variables(num_layers: int) -> List[str]:
    r"""Find the variables required for a certain number of layers in the
    input circuit and degree.

    Args:
        num_layers: Number of layers in the input circuit.

    Returns:
        variables: Variables required for the monomial basis.
    """
    return [f"λ_{i}" for i in range(1, num_layers + 1)]


def _create_variable_combinations(num_layers: int, degree: int):
    variables = _get_variables(num_layers)
    variable_combinations = []
    for i in range(degree, -1, -1):
        # Generate combinations for the current degree
        # Ranges from max degree, max degree -1, ...0.
        combos = list(combinations_with_replacement(variables, i))
        variable_combinations.append(combos)

    # return a flattened list
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
        monomial_basis: Monomial basis terms required for multivariate
            extrapolation upto max degree
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
):
    """Calculates the sample matrix required for extrapolation."""
    if degree < 1:
        raise ValueError(
            "Multinomial degree must be greater than or equal to 1."
        )
    if fold_multiplier < 1:
        raise ValueError("Fold multiplier must be greater than or equal to 1.")

    num_layers = _get_num_layers_without_measurements(input_circuit)

    scale_factor_vectors = _get_scale_factor_vectors(
        input_circuit, degree, fold_multiplier, num_chunks
    )

    monomial_terms = full_monomial_basis(num_layers, degree)
    if len(monomial_terms) != len(scale_factor_vectors):
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


def linear_combination_coefficients():
    """Finds the coefficients."""
