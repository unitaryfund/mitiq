# Copyright (C) 2020 Unitary Fund
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

"""Defines utilities for efficiently running collections of circuits generated
by error mitigation techniques to compute expectation values."""

import inspect
from typing import Any, Callable, Iterable, List, Sequence, Tuple, Union

import numpy as np

import cirq
from mitiq import QPROGRAM
from mitiq.conversions import (
    convert_to_mitiq,
    UnsupportedCircuitError,
    CircuitConversionError,
)
from mitiq.utils import _equal


class Collector:
    """Tool for efficiently scheduling/executing quantum programs and
    collecting the results.
    """

    def __init__(self, executor: Callable, max_batch_size: int = 100) -> None:
        """Initializes a Collector.

        Args:
            executor: A function which inputs a program and outputs a float,
                or inputs a sequence of programs and outputs a sequence of
                floats.
            max_batch_size: Maximum number of programs that can be sent in a
                single batch (if the executor is batched).
        """
        self._executor = executor
        self._can_batch = Collector.is_batched_executor(executor)
        self._max_batch_size = max_batch_size

        self._executed_programs: List[QPROGRAM] = []
        self._computed_results: List[float] = []

        self._calls_to_executor: int = 0

    @property
    def can_batch(self) -> bool:
        return self._can_batch

    @property
    def calls_to_executor(self) -> int:
        return self._calls_to_executor

    def run(self, circuits: Sequence[QPROGRAM]) -> List[float]:
        collection = CircuitCollection(circuits)
        to_run = collection.unique

        if not self._can_batch:
            for circuit in to_run:
                self._call_executor(circuit)

        else:
            if self._max_batch_size >= len(to_run):
                self._call_executor(to_run)

            else:
                stop = len(to_run)
                step = self._max_batch_size
                for i in range(int(np.ceil(stop / step))):
                    batch = to_run[i * step : (i + 1) * step]
                    self._call_executor(batch)

        # Expand computed results to all results using multiplicities.
        results = []
        for value, mult in zip(self._computed_results, collection.counts()):
            results += [value] * mult
        return results

    def _call_executor(
        self, to_run: Union[QPROGRAM, Sequence[QPROGRAM]]
    ) -> None:
        """Calls the executor on the input circuit(s) to run.

        Args:
            to_run: Circuit(s) to run.
        """
        # TODO: Make sure `to_run` is a List[QPROGRAM].
        result = self._executor(to_run)
        self._calls_to_executor += 1

        try:
            result = list(result)
            self._computed_results += result
            self._executed_programs += to_run
        except TypeError:
            self._computed_results.append(result)
            self._executed_programs.append(to_run)

    @staticmethod
    def is_batched_executor(executor: Callable) -> bool:
        """Returns True if the input function is recognized as a "batched
        executor", else False.

        The executor is detected as "batched" if and only if it is annotated
        with a return type that is one of the following:

            * Iterable[float]
            * List[float]
            * Sequence[float]
            * Tuple[float]
            * numpy.ndarray

        Batched executors can run several quantum programs in a single call.
        See below.

        Args:
            executor: A "serial executor" (1) or a "batched executor" (2).

                (1) A function which inputs a single `QPROGRAM` and outputs a
                single expectation value as a float.
                (2) A function which inputs a list of `QPROGRAM`s and outputs a
                list of expectation values (one for each `QPROGRAM`).

        Returns:
            True if the executor is detected as batched, else False.
        """
        executor_annotation = inspect.getfullargspec(executor).annotations

        return executor_annotation.get("return") in (
            List[float],
            Sequence[float],
            Tuple[float],
            Iterable[float],
            np.ndarray,
        )


class CircuitCollection:
    """A collection of circuits to execute."""

    def __init__(self, circuits: Sequence[QPROGRAM]) -> None:
        self._raw_circuits = circuits
        self._cirq_circuits = [
            convert_to_mitiq(circuit)[0] for circuit in circuits
        ]

        self._unique: List[cirq.Circuit] = []
        self._counts = {}

        for i, circ in enumerate(self._cirq_circuits):
            found = False
            for j, circuit in enumerate(self._unique):
                if _equal(circ, circuit):
                    self._counts[list(self._counts.keys())[j]] += 1
                    found = True
                    break

            if not found:
                self._unique.append(circ)
                self._counts[i] = 1

    @property
    def all(self) -> List[QPROGRAM]:
        return list(self._raw_circuits)

    @property
    def unique(self) -> List[QPROGRAM]:
        return [self._raw_circuits[i] for i in self._counts.keys()]

    def counts(self) -> List[int]:
        return list(self._counts.values())

    def unique_with_counts(self) -> List[Tuple[QPROGRAM, int]]:
        return [(self._raw_circuits[k], v) for k, v in self._counts.items()]

    def multiplicity_of(self, item: Any) -> int:
        try:
            item, _ = convert_to_mitiq(item)
        except (CircuitConversionError, UnsupportedCircuitError):
            return 0

        for i, circ in enumerate(self._cirq_circuits):
            if _equal(item, circ):
                return self._counts[i]
        return 0

    def __contains__(self, item: Any) -> bool:
        try:
            circuit, _ = convert_to_mitiq(item)
        except (CircuitConversionError, UnsupportedCircuitError):
            return False

        for circ in self._unique:
            if _equal(circuit, circ):
                return True
        return False
