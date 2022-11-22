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

from typing import Callable, Union

from functools import wraps
import numpy as np
import numpy.typing as npt

from mitiq._typing import QPROGRAM, MeasurementResult
from mitiq.executor.executor import Executor
from mitiq.observable.observable import Observable
from mitiq.rem.inverse_confusion_matrix import mitigate_measurements


def execute_with_rem(
    circuit: QPROGRAM,
    executor: Union[Executor, Callable[[QPROGRAM], MeasurementResult]],
    observable: Observable,
    *,
    inverse_confusion_matrix: npt.NDArray[np.float64],
) -> float:
    """Returns the readout error mitigated expectation value utilizing an
    inverse confusion matrix.

    Args:
        executor: A Mitiq executor that executes a circuit and returns the
            unmitigated ``MeasurementResult``.
        observable: Observable to compute the expectation value of (required).
        inverse_confusion_matrix: The inverse confusion matrix to apply to the
            probability vector estimated with noisy measurement results.

    Returns:
        The expectation value estimated with REM.
    """
    if not isinstance(executor, Executor):
        executor = Executor(executor)

    result = executor._run([circuit])
    noisy_result = result[0]
    if not isinstance(noisy_result, MeasurementResult):
        raise TypeError("Results are not of type MeasurementResult.")

    mitigated_result = mitigate_measurements(
        noisy_result, inverse_confusion_matrix
    )

    return observable._expectation_from_measurements([mitigated_result])


def mitigate_executor(
    executor: Callable[[QPROGRAM], MeasurementResult],
    observable: Observable,
    *,
    inverse_confusion_matrix: npt.NDArray[np.float64],
) -> Callable[[QPROGRAM], float]:
    """Returns a modified version of the input 'executor' which is
    error-mitigated with readout confusion inversion (RCI).

    Args:
        executor: A Mitiq executor that executes a circuit and returns the
            unmitigated ``MeasurementResult``.
        observable: Observable to compute the expectation value of (required).
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
            observable,
            inverse_confusion_matrix=inverse_confusion_matrix,
        )

    return new_executor


def rem_decorator(
    observable: Observable,
    *,
    inverse_confusion_matrix: npt.NDArray[np.float64],
) -> Callable[
    [Callable[[QPROGRAM], MeasurementResult]], Callable[[QPROGRAM], float]
]:
    """Decorator which adds an error-mitigation layer based on readout
    confusion inversion (RCI) to an executor function, i.e., a function
    which executes a quantum circuit with an arbitrary backend and returns
    a ``MeasurementResult``.

    Args:
        observable: Observable to compute the expectation value of (required).
        inverse_confusion_matrix: The inverse confusion matrix to apply to the
            probability vector estimated with noisy measurement results
            (required).

    Returns:
        The error-mitigating decorator to be applied to an executor function.
    """
    # NOTE: most decorators check for whether the decorator has been used
    #   without parenthesis, but that is not possible with this decorator
    #   since arguments are required.

    def decorator(
        executor: Callable[[QPROGRAM], MeasurementResult]
    ) -> Callable[[QPROGRAM], float]:
        return mitigate_executor(
            executor,
            observable,
            inverse_confusion_matrix=inverse_confusion_matrix,
        )

    return decorator
