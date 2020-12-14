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

from collections import Counter
import inspect
from typing import Callable, Iterable, List, Sequence, Tuple, Union

import numpy as np

import cirq
from mitiq import QPROGRAM
from mitiq.conversions import (
    convert_from_mitiq,
    convert_to_mitiq,
)


class Collector:
    """Tool for efficiently scheduling/executing quantum programs and
    collecting the results.
    """

    def __init__(self, executor: Callable, max_batch_size: int = 75) -> None:
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

        self._executed_circuits: List[QPROGRAM] = []
        self._computed_results: List[float] = []

        self._calls_to_executor: int = 0

    @property
    def can_batch(self) -> bool:
        return self._can_batch

    @property
    def calls_to_executor(self) -> int:
        return self._calls_to_executor

    def run(
        self, circuits: Sequence[QPROGRAM], force_run_all: bool = False
    ) -> List[float]:
        """Runs all input circuits using the least number of possible calls to
        the executor.

        Args:
            circuits: Sequence of circuits to execute using the executor.
            force_run_all: If True, force every circuit in the input sequence
            to be executed (if some are identical). Else, detects identical
            circuits and runs a minimal set.
        """
        if force_run_all:
            to_run = circuits
        else:
            # Convert to FrozenCircuits
            # Note: Assumes all circuits are the same type.
            _, conversion_type = convert_to_mitiq(circuits[0])
            circuits = [
                cirq.FrozenCircuit(
                    convert_to_mitiq(circ)[0].all_operations(),
                    strategy=cirq.InsertStrategy.EARLIEST,
                )
                for circ in circuits
            ]

            # Get the unique circuits and counts
            collection = Counter(circuits)
            to_run = [
                convert_from_mitiq(circ, conversion_type)
                for circ in collection.keys()
            ]
            counts = list(collection.values())

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
                    batch = to_run[i * step: (i + 1) * step]
                    self._call_executor(batch)

        # Expand computed results to all results using counts.
        if force_run_all:
            return self._computed_results
        results = []
        for value, mult in zip(self._computed_results, counts):
            results += [value] * mult
        return results

    def _call_executor(
        self, to_run: Union[QPROGRAM, Sequence[QPROGRAM]]
    ) -> None:
        """Calls the executor on the input circuit(s) to run.

        Args:
            to_run: Circuit(s) to run.
        """
        result = self._executor(to_run)
        self._calls_to_executor += 1

        try:
            result = list(result)
            self._computed_results += result
            self._executed_circuits += to_run
        except TypeError:
            self._computed_results.append(result)
            self._executed_circuits.append(to_run)

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
