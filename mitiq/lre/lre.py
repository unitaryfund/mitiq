# Copyright (C) Unitary Fund
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

"""Extrapolation methods for Layerwise Richardson Extrapolation (LRE)"""

from functools import wraps
from typing import Any, Callable, Optional, Union

import numpy as np
from cirq import Circuit

from mitiq import Observable, QuantumResult
from mitiq.lre import (
    multivariate_layer_scaling,
    multivariate_richardson_coefficients,
)
from mitiq.zne.scaling import fold_gates_at_random


def execute_with_lre(
    input_circuit: Circuit,
    executor: Callable[[Circuit], QuantumResult],
    degree: int,
    fold_multiplier: int,
    folding_method: Callable[
        [Union[Any], float], Union[Any]
    ] = fold_gates_at_random,
    num_chunks: Optional[int] = None,
    observable: Optional[Observable] = None,
) -> float:
    noise_scaled_circuits = multivariate_layer_scaling(
        input_circuit, degree, fold_multiplier, num_chunks, folding_method
    )
    linear_combination_coeffs = multivariate_richardson_coefficients(
        input_circuit, degree, fold_multiplier, num_chunks
    )

    lre_exp_values = []
    for scaled_circuit in noise_scaled_circuits:
        circ_exp_val = executor(scaled_circuit)
        lre_exp_values.append(circ_exp_val)

    # verify the linear combination coefficients and the calculated expectation
    # values have the same length
    assert len(lre_exp_values) == len(linear_combination_coeffs)

    return np.dot(lre_exp_values, linear_combination_coeffs)


def mitigate_executor(
    executor: Callable[[Circuit], QuantumResult],
    degree: int,
    fold_multiplier: int,
    folding_method: Callable[
        [Union[Any], float], Union[Any]
    ] = fold_gates_at_random,
    num_chunks: Optional[int] = None,
    observable: Optional[Observable] = None,
) -> Callable[[Circuit], float]:
    @wraps(executor)
    def new_executor(input_circuit: Circuit) -> float:
        return execute_with_lre(
            input_circuit,
            executor,
            degree,
            fold_multiplier,
            folding_method,
            num_chunks,
            observable,
        )

    return new_executor


def lre_decorator(
    degree: int,
    fold_multiplier: int,
    folding_method: Callable[[Circuit, float], Circuit] = fold_gates_at_random,
    num_chunks: Optional[int] = None,
    observable: Optional[Observable] = None,
) -> Callable[
    [Callable[[Circuit], QuantumResult]], Callable[[Circuit], float]
]:
    def decorator(
        executor: Callable[[Circuit], QuantumResult],
    ) -> Callable[[Circuit], float]:
        return mitigate_executor(
            executor,
            degree,
            fold_multiplier,
            folding_method,
            num_chunks,
            observable,
        )

    return decorator
