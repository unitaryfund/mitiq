# Copyright (C) Unitary Fund
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

"""Functions for layerwise folding of input circuits to allow for multivariate
extrapolation."""

import itertools
from copy import deepcopy

import cirq
import numpy as np

from mitiq.utils import _pop_measurements
from mitiq.zne.scaling.folding import _check_foldable


def _get_num_layers_without_measurements(input_circuit: cirq.Circuit) -> int:
    """Checks if the circuit has non-terminal measurements and returns the
    number of layers in the input circuit without the terminal measurements.

        Args:
            input_circuit: Checks whether this circuit is able to be folded.

        Raises:
            UnfoldableCircuitError:
                * If the circuit has intermediate measurements.
                * If the circuit has non-unitary channels which are not
                terminal measurements.

        Returns:
            num_layers: the number of layers in the input circuit without the
            terminal measurements.
    """
    circuit = deepcopy(input_circuit)
    _check_foldable(circuit)
    _pop_measurements(circuit)
    num_layers = len(circuit)
    return num_layers


def _get_scale_factor_vectors(
    input_circuit: cirq.Circuit, degree: int, fold_multiplier: int
) -> list[tuple[int]]:
    """Returns the patterned scale factor vectors required for multivariate
    extrapolation.

        Args:
            input_circuit: Circuit to be scaled
            degree: Degree of the multivariate polynomial
            fold_multiplier: Scaling gap required by unitary folding

        Returns:
            scale_factor_vectors: Multiple variations of scale factors for each
            layer in the input circuit
    """
    num_layers = _get_num_layers_without_measurements(input_circuit)

    # find the exponents of all the monomial terms required for the folding
    # pattern
    pattern_full = []
    for i in range(degree + 1):
        for j in itertools.combinations_with_replacement(range(num_layers), i):
            pattern = [0] * num_layers
            # get the monomial terms in graded lexicographic order
            for index in j:
                pattern[index] += 1
            # use fold multiplier on the folding pattern to figure out which
            # layers will be scaled
            pattern_full.append(tuple(fold_multiplier * np.array(pattern)))

    # get the scale factor vectors
    # the layers are scaled as 2n+1 for unitary folding
    return [
        tuple(2 * num_folds + 1 for num_folds in pattern)
        for pattern in pattern_full
    ]
