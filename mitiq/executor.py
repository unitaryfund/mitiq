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

"""Defines the interface that converts supported quantum circuits to executed
results which is called in a black-box manner by error mitigation techniques.
"""

import inspect
from typing import Any, Callable, Iterable, List, Sequence, Tuple, Union

import numpy as np

from cirq.study import TrialResult


MeasurementResult = TrialResult


class Executor:
    def __init__(
        self,
        circuit_to_expectation: Callable = None,
        circuit_to_measurement: Callable = None,
        measurement_to_expectation: Callable = None,
    ) -> None:
        """Constructs an Executor object to run circuits and return expectation
        values, optionally storing raw measurement results.

        Succinctly, the workflow is

        (a) Circuit(s) -> Expectation value(s) via ``circuit_to_expectation``

        or

        (b) Circuit(s) -> Measurements via ``circuit_to_measurement`` then
            Measurements -> Expectation value(s) via
            ``measurement_to_expectation``.

        If a function outputs more than one expectation value (is "batched"),
        it must be annotated with a return type that is one of the following:

        * Iterable[float]
        * List[float]
        * Sequence[float]
        * Tuple[float]
        * numpy.ndarray

        Otherwise, by default the function is assumed to output one expectation
        value (is "serial").

        Args:
            circuit_to_expectation: A function which inputs one or more
                circuits and outputs one or more expectation values.
            circuit_to_measurement: A function which inputs one or more
                circuits and outputs one or more measurement results.
            measurement_to_expectation: A function which inputs one or more
                measurement results and outputs one or more expectation values.
        """
        self._measurement_history: List[MeasurementResult] = []
        self._expectation_history: List[float] = []
        self._circuit_history: List[Any] = []

        self._supports_measurements = circuit_to_measurement is not None
        if circuit_to_measurement:
            self._execute = circuit_to_measurement
        else:
            self._execute = circuit_to_expectation
        self._measurement_to_expectation = measurement_to_expectation

        if not self._execute or (
            self._supports_measurements
            and not self._measurement_to_expectation
        ):
            raise ValueError(
                "Unable to compute expectation values with provided arguments."
                "Provide either `circuit_to_expectation` OR "
                "`circuit_to_measurement` AND `measurement_to_expectation`."
            )

        self._is_batched = self._can_batch()

    def _can_batch(self) -> bool:
        """Returns True if the Executor can batch execute circuits."""
        if self._supports_measurements:
            execute = self._measurement_to_expectation
        else:
            execute = self._execute

        annotation = inspect.getfullargspec(execute).annotations

        return annotation.get("return") in (
            List[float],
            Sequence[float],
            Tuple[float],
            Iterable[float],
            np.ndarray,
        )

    @property
    def circuit_history(self) -> List[Any]:
        return self._circuit_history

    @property
    def measurement_history(self) -> List[MeasurementResult]:
        return self._measurement_history

    @property
    def expectation_history(self) -> List[float]:
        return self._expectation_history

    @property
    def last_measurement(self) -> MeasurementResult:
        return self._measurement_history[-1]

    @property
    def last_expectation(self) -> Union[float, List[float]]:
        return self._expectation_history[-1]

    def execute(self, to_run: Any) -> Union[float, List[float]]:
        """Returns the floating point expectation value(s) obtained by
        executing the circuit(s) and stores the result in history.
        Stores measurement results if supported.

        Args:
            to_run: A single circuit or sequence of circuits to execute.
        """
        result = self._execute(to_run)

        if self._supports_measurements:
            self._measurement_history.append(result)
            self._expectation_history.append(
                self._measurement_to_expectation(result)
            )
        else:
            self._expectation_history.append(result)

        self._circuit_history.append(to_run)
        return self.last_expectation
