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

from typing import Callable, Union, List, Sequence
from copy import deepcopy
from functools import wraps
import numpy as np
import numpy.typing as npt

from mitiq import QPROGRAM, MeasurementResult
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

    executor_with_rem = mitigate_executor(
        executor, inverse_confusion_matrix=inverse_confusion_matrix
    )

    return executor_with_rem.evaluate(circuit, observable)[0]


def mitigate_executor(
    executor: Union[Executor, Callable[[QPROGRAM], MeasurementResult]],
    *,
    inverse_confusion_matrix: npt.NDArray[np.float64],
) -> Union[Executor, Callable[[QPROGRAM], MeasurementResult]]:
    """Returns a modified version of the input 'executor' which is
    error-mitigated with readout confusion inversion (RCI).

    Args:
        executor: A Mitiq executor that executes a circuit and returns the
            unmitigated ``MeasurementResult``.
        inverse_confusion_matrix: The inverse confusion matrix to apply to the
            probability vector estimated with noisy measurement results.

    Returns:
        The error-mitigated version of the input executor.
    """
    if not isinstance(executor, Executor):
        executor_obj = Executor(executor)
    else:
        executor_obj = deepcopy(executor)

    def post_run(
        results: Sequence[MeasurementResult],
    ) -> Sequence[MeasurementResult]:
        return [
            mitigate_measurements(res, inverse_confusion_matrix)
            for res in results
        ]

    executor_obj._post_run = post_run

    if isinstance(executor, Executor):
        new_executor = executor_obj

    elif not executor_obj.can_batch:

        @wraps(executor)
        def new_executor(circuit: QPROGRAM) -> MeasurementResult:
            return executor_obj._run([circuit])[0]

    elif executor_obj.can_batch:

        @wraps(executor)
        def new_executor(
            circuits: List[QPROGRAM],
        ) -> Sequence[MeasurementResult]:
            return executor_obj._run(circuits)

    return new_executor


def rem_decorator(
    *,
    inverse_confusion_matrix: npt.NDArray[np.float64],
) -> Callable[
    [Callable[[QPROGRAM], MeasurementResult]],
    Callable[[QPROGRAM], MeasurementResult],
]:
    """Decorator which adds an error-mitigation layer based on readout
    confusion inversion (RCI) to an executor function, i.e., a function
    which executes a quantum circuit with an arbitrary backend and returns
    a ``MeasurementResult``.

    Args:
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
    ) -> Callable[[QPROGRAM], MeasurementResult]:
        return mitigate_executor(
            executor,
            inverse_confusion_matrix=inverse_confusion_matrix,
        )

    return decorator
