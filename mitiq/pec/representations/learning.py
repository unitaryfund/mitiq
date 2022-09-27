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
import numpy.typing as npt
from scipy.optimize import minimize
from mitiq import QPROGRAM, Executor, Observable
from mitiq.cdr import generate_training_circuits
from mitiq.pec import execute_with_pec
from mitiq.pec.representations.biased_noise import (
    represent_operation_with_local_biased_noise,
)


def learn_depolarizing_noise_parameter(
    operations_to_learn: List[QPROGRAM],
    circuit: QPROGRAM,
    ideal_executor: Executor,
    noisy_executor: Executor,
    pec_kwargs: Dict["str", Any],
    num_training_circuits: int = 5,
    fraction_non_clifford: float = 0.2,
    training_random_state: np.random.RandomState = None,  # type: ignore
    epsilon0: float = 0.05,
    observable: Optional[Observable] = None,
    method: Optional[str] = "Nelder-Mead",
    **minimize_kwargs: Dict["str", Any],
) -> Tuple[bool, float]:
    r"""This function learns the depolarizing noise parameter (epsilon)
    associated to a set of input operations. The learning process is based on
    the execution of a set of training circuits on a noisy backend and on a
    classical simulator. The training circuits are near-Clifford approximations
    of the input circuit. A depolarizing noise model characterization is
    assumed.

    Args:
        operations_to_learn: The ideal operations to learn the noise model of.
        circuit: The full quantum program as defined by the user.
        ideal_executor: Simulates a circuit and returns
            a noiseless ``QuantumResult``.
        noisy_executor: Executes a circuit on a noisy backend
            and returns a ``QuantumResult``.
        pec_kwargs: Options to pass to ``execute_w_pec`` for the
            error-mitigated expectation value obtained from executing
            the training circuits.
        num_training_circuits: Number of near-Clifford circuits to be
            generated for training.
        fraction_non_clifford: The (approximate) fraction of non-Clifford
            gates in each training circuit.
        training_random_state: Seed for sampling the training circuits.
        epsilon0: Initial guess for noise strength.
        observable (optional): Observable to compute the expectation value of.
            If None, the ``executor`` must return an expectation value.
            Otherwise the `QuantumResult` returned by `executor` is used to
            compute the expectation of the observable.
        method: Type of optimizer. Must be an optimizer supported by
            ``scipy.optimize.minimize``.
        minimize_kwargs: Options to pass to the solver called by
            ``scipy.optimize.minizmize``.

    Returns:
        A flag indicating whether or not the optimizer exited successfully and
        the optimized noise strength epsilon.

    .. note:: Using this function may require some tuning. One of the main
        challenges is setting a good value of ``num_samples`` in the PEC
        options ``pec_kwargs``. Setting a small value of ``num_samples`` is
        typicallynecessary to obtain a reasonable execution time. On the other
        hand,using a number of PEC samples that is too small can result in a
        large statistical error, ultimately causing the optimization process to
        fail.
    """
    training_circuits = generate_training_circuits(
        circuit=circuit,
        num_training_circuits=num_training_circuits,
        fraction_non_clifford=fraction_non_clifford,
        random_state=training_random_state,
    )

    ideal_values = np.array(
        [ideal_executor.evaluate(t, observable) for t in training_circuits]
    )

    def depolarizing_noise_loss_function(
        epsilon: List[int],
        operations_to_mitigate: List[QPROGRAM],
        training_circuits: List[QPROGRAM],
        ideal_values: npt.NDArray[np.float64],
        noisy_executor: Executor,
        pec_kwargs: Dict["str", Any],
        observable: Optional[Observable] = None,
    ) -> float:
        return biased_noise_loss_function(
            np.array([epsilon[0], 0]),
            operations_to_mitigate,
            training_circuits,
            ideal_values,
            noisy_executor,
            pec_kwargs,
            observable,
        )

    result = minimize(
        depolarizing_noise_loss_function,
        epsilon0,
        args=(
            operations_to_learn,
            training_circuits,
            ideal_values,
            noisy_executor,
            pec_kwargs,
            observable,
        ),
        method=method,
        **minimize_kwargs,
    )
    x_result = result.x
    success = cast(bool, result.success)
    epsilon_opt = cast(float, x_result[0])

    return success, epsilon_opt


def biased_noise_loss_function(
    params: npt.NDArray[np.float64],
    operations_to_mitigate: List[QPROGRAM],
    training_circuits: List[QPROGRAM],
    ideal_values: npt.NDArray[np.float64],
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
            ``QuantumResult``.
        pec_kwargs: Options to pass to ``execute_w_pec`` for the
            error-mitigated expectation value obtained from executing the
            training circuits.
        observable (optional): Observable to compute the expectation value of.
            If None, the ``executor`` must return an expectation value.
            Otherwise the ``QuantumResult`` returned by ``executor`` is used to
            compute the expectation of the observable.

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

    return np.mean(
        abs(mitigated_values.reshape(-1, 1) - ideal_values.reshape(-1, 1)) ** 2
    )
