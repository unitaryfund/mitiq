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

from typing import Callable, Optional, Union

from functools import wraps
import numpy as np

from mitiq._typing import QPROGRAM, MeasurementResult
from mitiq.executor.executor import Executor
from mitiq.observable.observable import Observable


def to_probability_vector(measurement: np.ndarray) -> np.ndarray:
    """Convert a raw measurement to a probability vector.

    Args:
        measurement: A single measurement.
    """
    index = int("".join([str(m) for m in measurement]), base=2)
    pv = np.zeros((2 ** measurement.shape[0]), dtype=np.uint8)
    pv[index] = 1
    return pv


def sample_probability_vector(probability_vector: np.ndarray) -> np.ndarray:
    """Sample a probability vector and returns a bitstring.

    Args:
        probability_vector: A probability vector.
    """
    result = np.random.choice(
        len(probability_vector), size=1, p=probability_vector
    )[0]
    value = [int(i) for i in bin(result)[2:]]
    return np.array(value, dtype=np.uint8)


def mitigate_measurements(
    noisy_result: MeasurementResult,
    inverse_confusion_matrix: np.ndarray,
) -> MeasurementResult:
    """Applies the inverse confusion matrix against the noisy measurement
    result and returns the adjusted measurements.

    Args:
        noisy_results: The unmitigated ``MeasurementResult``.
        inverse_confusion_matrix: The inverse confusion matrix to apply to the
            probability vector estimated with noisy measurement results.
    """
    assert noisy_result.qubit_indices is not None
    num_qubits = len(noisy_result.qubit_indices)
    assert inverse_confusion_matrix.shape == (2**num_qubits, 2**num_qubits)

    empirical_prob_dist = np.apply_along_axis(
        to_probability_vector, 1, noisy_result.asarray
    )

    adjusted_prob_dist = (inverse_confusion_matrix @ empirical_prob_dist.T).T

    adjusted_result = np.apply_along_axis(
        sample_probability_vector, 1, adjusted_prob_dist
    )

    result = MeasurementResult(adjusted_result, noisy_result.qubit_indices)
    return result


def execute_with_rem(
    circuit: QPROGRAM,
    executor: Union[Executor, Callable[[QPROGRAM], MeasurementResult]],
    inverse_confusion_matrix: np.ndarray,
    *,
    observable: Optional[Observable] = None,
) -> float:
    """Returns the readout error mitigated expectation value utilizing an
    inverse confusion matrix.

    Args:
        executor: A Mitiq executor that executes a circuit and returns the
            unmitigated ``MeasurementResult``.
        observable: Observable to compute the expectation value of.
        inverse_confusion_matrix: The inverse confusion matrix to apply to the
            probability vector estimated with noisy measurement results.
    """
    if not isinstance(executor, Executor):
        executor = Executor(executor)

    result = executor._run([circuit])
    noisy_result = result[0]
    assert isinstance(noisy_result, MeasurementResult)

    mitigated_result = mitigate_measurements(
        noisy_result, inverse_confusion_matrix
    )
    assert observable is not None
    return observable._expectation_from_measurements([mitigated_result])


def mitigate_executor(
    executor: Callable[[QPROGRAM], MeasurementResult],
    inverse_confusion_matrix: np.ndarray,
    *,
    observable: Optional[Observable] = None,
) -> Callable[[QPROGRAM], float]:
    """Returns a modified version of the input 'executor' which is
    error-mitigated with readout confusion inversion (RCI).

    Args:
        executor: A Mitiq executor that executes a circuit and returns the
            unmitigated ``MeasurementResult``.
        observable: Observable to compute the expectation value of.
        inverse_confusion_matrix: The inverse confusion matrix to apply to the
            probability vector estimated with noisy measurement results.

    Returns:
        The error-mitigated version of the input executor.
    """

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
    inverse_confusion_matrix: np.ndarray,
    *,
    observable: Optional[Observable] = None,
) -> Callable[
    [Callable[[QPROGRAM], MeasurementResult]], Callable[[QPROGRAM], float]
]:
    """Decorator which adds an error-mitigation layer based on readout
    confusion inversion (RCI) to an executor function, i.e., a function
    which executes a quantum circuit with an arbitrary backend and returns
    a ``MeasurementResult``.

    Args:
        observable: Observable to compute the expectation value of.
        inverse_confusion_matrix: The inverse confusion matrix to apply to the
            probability vector estimated with noisy measurement results.

    Returns:
        The error-mitigating decorator to be applied to an executor function.
    """
    # Raise an error if the decorator is used without parenthesis
    if callable(inverse_confusion_matrix):
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
