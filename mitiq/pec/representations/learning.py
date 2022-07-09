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

from typing import cast, Optional, Dict, Any, List, Tuple
import numpy as np
from scipy.optimize import minimize
from mitiq import QPROGRAM, Executor, Observable
from mitiq.cdr import generate_training_circuits
from mitiq.pec import execute_with_pec
from mitiq.pec.representations.biased_noise import (
    represent_operation_with_local_biased_noise,
)


def learn_biased_noise_parameters(
    operations_to_learn: List[QPROGRAM],
    circuit: QPROGRAM,
    ideal_executor: Executor,
    noisy_executor: Executor,
    pec_kwargs: Dict["str", Any],
    num_training_circuits: int = 10,
    fraction_non_clifford: float = 0.5,
    epsilon0: float = 0,
    eta0: float = 1,
    observable: Optional[Observable] = None,
    **minimize_kwargs: Dict["str", Any],
) -> Tuple[float, float, bool]:
    r"""This function learns optimal noise parameters for a set of input
    operations.The learning process is based on the execution of a set of
    training circuits on a noisy backend and on a classical simulator. The
    training circuits are near-Clifford approximations of the input circuit.
    A biased nose model characterized by two parameters (epsilon and eta)
    is assumed.

    Args:
        operations_to_learn: The ideal operations to learn the noise model of.
        circuit: The full quantum program as defined by the user.
        ideal_executor: Simulates a circuit and returns
        a noiseless `QuantumResult`.
        noisy_executor: Executes a circuit on a noisy backend
        and returns a `QuantumResult`.
        num_training_circuits: Number of near-Clifford circuits to be
            generated for training.
        epsilon0: Initial guess for noise strength.
        eta0: Initial guess for noise bias.
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
        fraction_non_clifford=fraction_non_clifford,
        method_select="uniform",
        method_replace="closest",
    )

    ideal_values = np.array(
        [ideal_executor.evaluate(t, observable) for t in training_circuits]
    )

    x0 = np.array(
        [epsilon0, eta0]
    )  # initial parameter values for optimization
    result = minimize(
        biased_noise_loss_function,
        x0,
        args=(
            operations_to_learn,
            training_circuits,
            ideal_values,
            noisy_executor,
            pec_kwargs,
            observable,
        ),
        method="BFGS",
        **minimize_kwargs,
    )
    x_result = result.x
    epsilon_opt = cast(float, x_result[0])
    eta_opt = cast(float, x_result[1])
    success = cast(bool, result.success)
    return epsilon_opt, eta_opt, success


def biased_noise_loss_function(
    params: np.ndarray,
    operations_to_mitigate: List[QPROGRAM],
    training_circuits: List[QPROGRAM],
    ideal_values: np.ndarray,
    noisy_executor: Executor,
    pec_kwargs: Dict["str", Any],
    observable: Optional[Observable] = None,
) -> float:
    r"""Loss function for optimizing quasi-probability representations
    assuming a biased noise model depending on two real parameters.

    Args:
        params: Array of optimization parameters epsilon
            (local noise strength) and eta (noise bias between reduced
            dephasing and depolarizing channels).
        operations_to_mitigate: List of ideal operations to be represented by
            a combination of noisy operations.
        training_circuits: List of training circuits for generating the
            error-mitigated expectation values.
        ideal_values: Expectation values obtained by noiseless simulations.
        noisy_executor: Executes the circuit with noise and returns a
            `QuantumResult`.
        pec_kwargs: Options to pass to `execute_w_pec` for the error-mitigated
            expectation value obtained from executing the training circuits.
        observable (optional): Observable to compute the expectation value of.
            If None, the `executor` must return an expectation value. Otherwise
            the `QuantumResult` returned by `executor` is used to compute the
            expectation of the observable.

    Returns: Mean squared error between the error-mitigated values and
        the ideal values, over the training set.
    """
    epsilon = params[0]
    eta = params[1]
    representations = [
        represent_operation_with_local_biased_noise(
            operation,
            epsilon,
            eta,
        )
        for operation in operations_to_mitigate
    ]
    mitigated_values = np.array(
        [
            execute_with_pec(
                circuit=training_circuit,
                observable=observable,
                executor=noisy_executor,
                representations=representations,
                full_output=False,
                **pec_kwargs,
            )
            for training_circuit in training_circuits
        ]
    )

    return np.sum((mitigated_values - ideal_values) ** 2) / len(
        training_circuits
    )
