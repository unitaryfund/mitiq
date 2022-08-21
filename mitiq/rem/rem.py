# Copyright (C) 2021 Unitary Fund
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

"""Readout Confusion Inversion."""

from typing import Callable, Iterable, List, Optional, Sequence, Tuple, Union

from functools import partial, wraps
import numpy as np

from cirq.sim import sample_state_vector as sample_prob_dist
from cirq.qis.states import to_valid_state_vector as to_prob_dist

from .measurement_result import MeasurementResult

MatrixLike = Union[
    np.ndarray,
    Iterable[np.ndarray],
    List[np.ndarray],
    Sequence[np.ndarray],
    Tuple[np.ndarray],
]


def execute_with_rem(
    circuit: 'QPROGRAM',
    executor: Union['Executor', Callable[['QPROGRAM'], MeasurementResult]],
    inverse_confusion_matrix: MatrixLike,
    *,
    observable: Optional['Observable'] = None,
) -> float:
    """Returns the readout error mitigated expectation value utilizing an
    inverse confusion matrix that is computed by running the quantum program
    `circuit` with the executor function.

    Args:
        executor: A Mitiq executor that executes a circuit and returns the
            unmitigated ``MeasurementResult``.
        observable: Observable to compute the expectation value of.
        inverse_confusion_matrix: The inverse confusion matrix to apply to the
            probability vector that represents the noisy measurement results.
    """
    from mitiq.executor import Executor  # Avoid circular import.

    if not isinstance(executor, Executor):
        executor = Executor(executor)

    qubits = list(circuit.all_qubits())

    result = executor._run([circuit])
    noisy_result = result[0]
    assert isinstance(noisy_result, MeasurementResult)

    measurement_to_prob_dist = partial(
        to_prob_dist, num_qubits=len(qubits)
    )

    empirical_prob_dist = np.apply_along_axis(
        measurement_to_prob_dist, 1, noisy_result.asarray
    )

    adjusted_prob_dist = (
        inverse_confusion_matrix @ empirical_prob_dist.T
    ).T

    prob_dist_to_measurement = partial(
        sample_prob_dist, indices=noisy_result.qubit_indices
    )

    adjusted_result = np.apply_along_axis(
        prob_dist_to_measurement, 1, adjusted_prob_dist
    ).squeeze()

    result = MeasurementResult(adjusted_result, noisy_result.qubit_indices)
    return observable._expectation_from_measurements([result])


def mitigate_executor(
    executor: Callable[['QPROGRAM'], MeasurementResult],
    inverse_confusion_matrix: MatrixLike,
    *,
    observable: Optional['Observable'] = None,
) -> Callable[['QPROGRAM'], float]:
    """Returns a modified version of the input 'executor' which is
    error-mitigated with readout confusion inversion (RCI).

    Args:
        executor: A Mitiq executor that executes a circuit and returns the
            unmitigated ``MeasurementResult``.
        observable: Observable to compute the expectation value of.
        inverse_confusion_matrix: The inverse confusion matrix to apply to the
            probability vector that represents the noisy measurement results.

    Returns:
        The error-mitigated version of the input executor.
    """
    from mitiq._typing import QPROGRAM  # Avoid circular import.

    @wraps(executor)
    def new_executor(qp: QPROGRAM) -> float:
        return execute_with_rem(
            qp,
            executor,
            inverse_confusion_matrix,
            observable=observable,
        )

    return new_executor


def rem_decorator(
    inverse_confusion_matrix: MatrixLike,
    *,
    observable: Optional['Observable'] = None,
) -> Callable[
    [Callable[['QPROGRAM'], MeasurementResult]], Callable[['QPROGRAM'], float]
]:
    """Decorator which adds an error-mitigation layer based on readout
    confusion inversion (RCI) to an executor function, i.e., a function
    which executes a quantum circuit with an arbitrary backend and returns
    a ``MeasurementResult``.

    Args:
        observable: Observable to compute the expectation value of.
        inverse_confusion_matrix: The inverse confusion matrix to apply to the
            probability vector that represents the noisy measurement results.

    Returns:
        The error-mitigating decorator to be applied to an executor function.
    """
    from mitiq._typing import QPROGRAM  # Avoid circular import.

    # Raise an error if the decorator is used without parenthesis
    if callable(observable):
        raise TypeError(
            "Decorator must be used with parentheses (i.e., @rem_decorator()) "
            "even if no explicit arguments are passed."
        )

    def decorator(
        executor: Callable[[QPROGRAM], MeasurementResult]
    ) -> Callable[[QPROGRAM], float]:
        return mitigate_executor(
            executor,
            inverse_confusion_matrix,
            observable=observable,
        )

    return decorator
