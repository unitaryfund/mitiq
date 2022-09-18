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

from typing import Optional, Dict, Any, List
import numpy as np
from scipy.optimize import minimize
from cirq import (
    MixedUnitaryChannel,
    I,
    X,
    Y,
    Z,
    Circuit,
    ops,
    unitary,
)
from mitiq import QPROGRAM, Executor, Observable
from mitiq.interface import convert_to_mitiq
from mitiq.interface.mitiq_cirq import compute_density_matrix
from mitiq.cdr import generate_training_circuits
from mitiq.pec import execute_with_pec
from mitiq.pec.representations.biased_noise import (
    represent_operation_with_local_biased_noise,
)


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


def _unmitigated_loss_function(
    epsilon_guess: np.ndarray,
    noisy_executor: Executor,
    training_circuits: List[QPROGRAM],
    observable: Optional[Observable] = None,
) -> float:
    def ideal_execute(circ: Circuit) -> np.ndarray:
        return compute_density_matrix(circ, noise_level=(0.0,))

    def noise_model_execute(circ: Circuit) -> np.ndarray:
        circuit = convert_to_mitiq(circ)[0]
        noisy_circ = circuit.with_noise(
            biased_noise_channel(epsilon=epsilon_guess[0], eta=0)
        )
        return ideal_execute(noisy_circ)

    noise_model_executor = Executor(noise_model_execute)
    noise_model_values = np.array(
        [
            noise_model_executor.evaluate(training_circuit, observable)
            for training_circuit in training_circuits
        ]
    )

    noisy_values = np.array(
        [
            noisy_executor.evaluate(training_circuit, observable)
            for training_circuit in training_circuits
        ]
    )

    return np.sum(
        abs(noise_model_values.reshape(-1, 1) - noisy_values.reshape(-1, 1))
        ** 2
    ) / len(training_circuits)


def biased_noise_channel(epsilon: float, eta: float) -> MixedUnitaryChannel:
    a = 1 - epsilon
    b = epsilon * (3 * eta + 1) / (3 * (eta + 1))
    c = epsilon / (3 * (eta + 1))

    mix = [
        (a, unitary(I)),
        (b, unitary(Z)),
        (c, unitary(X)),
        (c, unitary(Y)),
    ]
    return ops.MixedUnitaryChannel(mix)


def learn_noise_parameters_from_unmitigated_data(
    circuit: QPROGRAM,
    epsilon0: float,
    noisy_executor: Executor,
    num_training_circuits: int = 5,
    fraction_non_clifford: float = 0.2,
    training_random_state: np.random.RandomState = None,
    observable: Optional[Observable] = None,
) -> float:
    training_circuits = generate_training_circuits(
        circuit,
        num_training_circuits=num_training_circuits,
        fraction_non_clifford=fraction_non_clifford,
        random_state=training_random_state,
    )
    result = minimize(
        fun=_unmitigated_loss_function,
        x0=epsilon0,
        args=(noisy_executor, training_circuits, observable),
        method="Nelder-Mead",
    )
    return result.success, result.x[0]
