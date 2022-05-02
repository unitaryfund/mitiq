# Copyright (C) 2022 Unitary Fund
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
"""Function to calculate parameters for biased noise model via a
learning-based technique."""

from typing import Callable, Optional, List
import numpy as np
from scipy.optimize import minimize
from mitiq import QPROGRAM, QuantumResult, Observable
from mitiq.cdr import generate_training_circuits
from mitiq.pec import execute_with_pec
from mitiq.pec.representations.biased_noise import (
    represent_operation_with_local_biased_noise,
)


def learn_noise_parameters(
    operation: QPROGRAM,
    circuit: QPROGRAM,
    ideal_executor: Callable[[QPROGRAM], QuantumResult],
    noisy_executor: Callable[[QPROGRAM], QuantumResult],
    num_training_circuits: int = 10,
    epsilon0: float = 0,
    eta0: float = 1,
    observable: Optional[Observable] = None,
):
    r"""Loss function: optimize the quasiprobability representation using
    the method of least squares

    Args:
        operation: ideal operation to be represented by a (learning-optmized)
            combination of noisy operations.
        circuit: the full quantum program as defined by the user.
        ideal_executor:
        noisy_executor:
        num_training_circuits: number of Clifford circuits to be generated for
            training data.
        epsilon0: initial guess for noise strength.
        eta0: initial guess for noise bias.
        observable (optional): a quantum observable typically used to compute
            its mitigated expectation value.

    Returns:
        Optimized noise strength epsilon and noise bias eta.
    """
    training_circuits = generate_training_circuits(
        circuit=circuit,
        num_training_circuits=num_training_circuits,
        fraction_non_clifford=0,
        method_select="uniform",
        method_replace="closest",
    )
    ideal_values = []
    for training_circuit in training_circuits:
        ideal_values.append(
            observable.expectation(training_circuit, ideal_executor).real
        )

    x0 = [epsilon0, eta0]  # initial parameter values for optimization
    result = minimize(
        loss_function,
        x0,
        args=(operation, circuit, ideal_values, noisy_executor),
        method="BFGS",
    )
    x_result = result.x
    epsilon = x_result[0]
    eta = x_result[1]

    return epsilon, eta


def loss_function(
    epsilon,
    eta,
    operation: QPROGRAM,
    circuit: QPROGRAM,
    ideal_values: List[np.ndarray],
    noisy_executor: Callable[[QPROGRAM], QuantumResult],
    observable: Optional[Observable] = None,
) -> float:
    r"""Loss function for optimizing the quasiprobability representation using
    the method of least squares
    Args:
        epsilon: local noise strength epsilon, an optimization parameter
        eta: noise bias between reduced dephasing and depolarizing
            channels, an optimization parameter
        operation: ideal operation to be represented by a (learning-optmized)
            combination of noisy operations
        ideal_values: expectation values obtained by simulations run on the
                    Clifford training circuits
    Returns: Square of the difference between the error-mitigated value and
        the ideal value, over the training set
    """
    representations = represent_operation_with_local_biased_noise(
        operation,
        epsilon,
        eta,
    )
    mitigated_value = execute_with_pec(
        circuit=circuit,
        observable=observable,
        executor=noisy_executor,
        representations=representations,
    )

    num_train = len(ideal_values)
    return (
        sum((mitigated_value * np.ones(num_train) - ideal_values) ** 2)
        / num_train
    )
