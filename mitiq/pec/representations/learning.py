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

from typing import Optional, List
import numpy as np
from scipy.optimize import minimize
from mitiq import QPROGRAM, Executor, Observable
from mitiq.cdr import generate_training_circuits
from mitiq.pec import execute_with_pec
from mitiq.pec.representations.biased_noise import (
    represent_operation_with_local_biased_noise,
)


def learn_biased_noise_parameters(
    operation: QPROGRAM,
    circuit: QPROGRAM,
    ideal_executor: Executor,
    noisy_executor: Executor,
    num_training_circuits: int = 10,
    epsilon0: float = 0,
    eta0: float = 1,
    observable: Optional[Observable] = None,
) -> List[float]:
    r"""Loss function: optimize the quasiprobability representation using
    the method of least squares

    Args:
        operation: ideal operation to be represented by a (learning-optmized)
            combination of noisy operations.
        circuit: the full quantum program as defined by the user.
        ideal_executor: Executes the ideal circuit and returns a
            `QuantumResult`.
        noisy_executor: Executes the noisy circuit and returns a
            `QuantumResult`.
        num_training_circuits: number of Clifford circuits to be generated for
            training data.
        epsilon0: initial guess for noise strength.
        eta0: initial guess for noise bias.
        observable (optional): Observable to compute the expectation value of.
            If None, the `executor` must return an expectation value. Otherwise
            the `QuantumResult` returned by `executor` is used to compute the
            expectation of the observable.

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
    ideal_values = np.array(
        ideal_executor.evaluate(training_circuits, observable)
    )

    x0 = [epsilon0, eta0]  # initial parameter values for optimization
    result = minimize(
        biased_noise_loss_function,
        x0,
        args=(
            operation,
            circuit,
            ideal_values,
            noisy_executor,
            num_training_circuits,
            observable,
        ),
        method="BFGS",
    )
    x_result = result.x
    epsilon = x_result[0]
    eta = x_result[1]

    return [epsilon, eta]


def biased_noise_loss_function(
    epsilon: float,
    eta: float,
    operation: QPROGRAM,
    circuit: QPROGRAM,
    ideal_values: np.ndarray,
    noisy_executor: Executor,
    num_training_circuits: int,
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
    representations = [
        represent_operation_with_local_biased_noise(
            operation,
            epsilon,
            eta,
        )
    ]
    mitigated = execute_with_pec(
        circuit=circuit,
        observable=observable,
        executor=noisy_executor,
        representations=representations,
    )
    
    if mitigated is float: 
        mitigated_value = mitigated
        
    else:
        mitigated_value = mitigated[0]

    return (
        sum(
            (mitigated_value * np.ones(num_training_circuits) - ideal_values)
            ** 2
        )
        / num_training_circuits
    )
