# Copyright (C) Unitary Fund
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

"""Functions for layerwise folding of input circuits to allow for multivariate
extrapolation."""

import itertools
from copy import deepcopy
from typing import Any, Callable, List, Optional, Tuple

import numpy as np
from cirq import Circuit

from mitiq import QPROGRAM
from mitiq.utils import _append_measurements, _pop_measurements
from mitiq.zne.scaling import fold_gates_at_random
from mitiq.zne.scaling.folding import _check_foldable


def _get_num_layers_without_measurements(input_circuit: Circuit) -> int:
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


def _get_chunks(
    input_circuit: Circuit, num_chunks: Optional[int] = None
) -> List[Circuit]:
    """Splits a circuit into approximately equal chunks.

    Adapted from:
    https://stackoverflow.com/questions/2130016/splitting-a-list-into-n-parts-of-approximately-equal-length

        Args:
            input_circuit: Circuit of interest.
            num_chunks: Number of desired approximately equal chunks
            * when num_chunks == num_layers, the original circuit is returned
            * when num_chunks == 1, the entire circuit is chunked into 1 layer


        Returns:
            split_circuit: Circuit of interest split into approximately equal
            chunks
    """
    num_layers = _get_num_layers_without_measurements(input_circuit)
    if num_chunks is None:
        num_chunks = num_layers

    if num_chunks == 0:
        raise ValueError("The number of chunks should be >= to 1.")

    if num_chunks > num_layers:
        raise ValueError(
            "Number of chunks > the number of layers in the circuit."
        )

    k, m = divmod(num_layers, num_chunks)
    split_circuit = (
        input_circuit[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)]
        for i in range(num_chunks)
    )
    return list(split_circuit)


def _get_scale_factor_vectors(
    input_circuit: Circuit,
    degree: int,
    fold_multiplier: int,
    num_chunks: Optional[int] = None,
) -> List[Tuple[Any, ...]]:
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

    circuit_chunks = _get_chunks(input_circuit, num_chunks)
    num_layers = len(circuit_chunks)

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


def multivariate_layer_scaling(
    input_circuit: Circuit,
    degree: int,
    fold_multiplier: int,
    num_chunks: Optional[int] = None,
    folding_method: Callable[
        [QPROGRAM, float], QPROGRAM
    ] = fold_gates_at_random,
) -> List[Circuit]:
    """Defines the noise scaling function required for Layerwise Richardson
    Extrapolation."""
    circuit_copy = deepcopy(input_circuit)
    terminal_measurements = _pop_measurements(circuit_copy)

    scaling_pattern = _get_scale_factor_vectors(
        circuit_copy, degree, fold_multiplier, num_chunks
    )

    chunks = _get_chunks(circuit_copy, num_chunks)

    multiple_folded_circuits = []
    for scale_factor_vector in scaling_pattern:
        folded_circuit = Circuit()
        for chunk, scale_factor in zip(chunks, scale_factor_vector):
            if scale_factor == 1:
                folded_circuit += chunk
            else:
                chunks_circ = Circuit(chunk)
                folded_chunk_circ = folding_method(chunks_circ, scale_factor)
                folded_circuit += folded_chunk_circ
        _append_measurements(folded_circuit, terminal_measurements)
        multiple_folded_circuits.append((folded_circuit))

    return multiple_folded_circuits
