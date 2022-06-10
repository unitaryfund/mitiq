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
from cirq import Circuit, LineQubit, Gate
from mitiq import QPROGRAM, Executor, Observable
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
    r"""Loss function for optimizing the quasiprobability representation using
    the method of least squares

    Args:
        params: array of optimization parameters epsilon
            (local noise strength) and eta (noise bias between reduced
            dephasing and depolarizing
            channels)
        operation_to_mitigate: list of ideal operations to be represented by a
            (learning-optmized) combination of noisy operations
        training_circuits: list of training circuits for generating the
            error-mitigated expectation values
        ideal_values: expectation values obtained by simulations run on the
            Clifford training circuit
        noisy_executor: Executes the circuit with noise and returns a
            `QuantumResult`.

        observable (optional): Observable to compute the expectation value of.
            If None, the `executor` must return an expectation value. Otherwise
            the `QuantumResult` returned by `executor` is used to compute the
            expectation of the observable.

    Returns: Square of the difference between the error-mitigated values and
        the ideal values, over the training set
    """
    epsilon = params[0]
    eta = params[1]
    representations = [
        represent_operation_with_local_biased_noise(
            Circuit(operation),
            epsilon,
            eta,
        ) for operation in operations_to_mitigate
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
