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
from dataclasses import dataclass
import inspect
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np

from cirq.linalg import partial_trace
from mitiq import QPROGRAM, QuantumResult

from mitiq.observable.observable import Observable
from mitiq.rem.measurement_result import MeasurementResult
from mitiq.interface import convert_from_mitiq, convert_to_mitiq


DensityMatrixLike = [
    np.ndarray,
    Iterable[np.ndarray],
    List[np.ndarray],
    Sequence[np.ndarray],
    Tuple[np.ndarray],
]
FloatLike = [
    float, Iterable[float], List[float], Sequence[float], Tuple[float]
]
MeasurementResultLike = [
    MeasurementResult,
    Iterable[MeasurementResult],
    List[MeasurementResult],
    Sequence[MeasurementResult],
    Tuple[MeasurementResult],
]


@dataclass(frozen=True)
class ExecutedResult:
    target_circuit: QPROGRAM
    observable: Observable
    executed_circuits: List[QPROGRAM]
    quantum_results: List[QuantumResult]
    computed_result: float


class Executor:
    """Tool for efficiently scheduling/executing quantum programs and
    collecting the results.
    """

    def __init__(
        self,
        executor: Callable[[Union[QPROGRAM, Sequence[QPROGRAM]]], Any],
        max_batch_size: int = 75,
    ) -> None:
        """Initializes an Executor.

        Args:
            executor: A function which inputs a program and outputs a float,
                or inputs a sequence of programs and outputs a sequence of
                floats.
            max_batch_size: Maximum number of programs that can be sent in a
                single batch (if the executor is batched).
        """
        self._executor = executor

        executor_annotation = inspect.getfullargspec(executor).annotations
        self._executor_return_type = executor_annotation.get("return")
        self._max_batch_size = max_batch_size

        self._executed_circuits: List[QPROGRAM] = []
        self._quantum_results: List[QuantumResult] = []
        self._executed_results: Dict[int, ExecutedResult]

        self._calls_to_executor: int = 0

    @property
    def can_batch(self) -> bool:
        return self._executor_return_type in (
            BatchedType[T]  # type: ignore[index]
            for BatchedType in [Iterable, List, Sequence, Tuple]
            for T in QuantumResult.__args__  # type: ignore[attr-defined]
        )

    @property
    def executed_circuits(self) -> List[QPROGRAM]:
        return self._executed_circuits

    @property
    def quantum_results(self) -> List[QuantumResult]:
        return self._quantum_results

    @property
    def calls_to_executor(self) -> int:
        return self._calls_to_executor

    def evaluate(
        self,
        circuits: Union[QPROGRAM, List[QPROGRAM]],
        observable: Optional[Observable] = None,
        force_run_all: bool = False,
        **kwargs: Any,
    ) -> List[complex]:
        if not isinstance(circuits, List):
            circuits = [circuits]

        # Get all required circuits to run.
        if (
            observable is not None
            and self._executor_return_type in MeasurementResultLike
        ):
            all_circuits = [
                circuit_with_measurements
                for circuit in circuits
                for circuit_with_measurements in observable.measure_in(circuit)
            ]
            result_step = observable.ngroups
        else:
            all_circuits = circuits
            result_step = 1

        # Run all required circuits.
        all_results = self._run(all_circuits, force_run_all, **kwargs)

        # Parse the results.
        if self._executor_return_type in FloatLike:
            results = all_results
        elif self._executor_return_type in DensityMatrixLike:
            results = []

            for density_matrix in all_results:
                # TODO: Make the following codeblock a function somewhere.
                observable_matrix = observable.matrix()

                if density_matrix.shape != observable_matrix.shape:
                    nqubits = int(np.log2(density_matrix.shape[0]))
                    density_matrix = partial_trace(
                        np.reshape(density_matrix, newshape=[2, 2] * nqubits),
                        keep_indices=observable.qubit_indices,
                    ).reshape(observable_matrix.shape)

                results.append(np.trace(density_matrix @ observable_matrix))
        elif self._executor_return_type in MeasurementResultLike:
            results = [
                observable._expectation_from_measurements(
                    all_results[i : i + result_step]
                )
                for i in range(len(all_results) // result_step)
            ]
        else:
            raise ValueError

        return results

    def _run(
        self,
        circuits: Sequence[QPROGRAM],
        force_run_all: bool = False,
        **kwargs: Any,
    ) -> List[QuantumResult]:
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
            # Make circuits hashable.
            # Note: Assumes all circuits are the same type.
            _, conversion_type = convert_to_mitiq(circuits[0])
            hashable_circuits = [
                convert_to_mitiq(circ)[0].freeze() for circ in circuits
            ]

            # Get the unique circuits and counts
            collection = Counter(hashable_circuits)
            to_run = [
                convert_from_mitiq(circ.unfreeze(), conversion_type)
                for circ in collection.keys()
            ]

        if not self.can_batch:
            for circuit in to_run:
                self._call_executor(circuit, **kwargs)

        else:
            stop = len(to_run)
            step = self._max_batch_size
            for i in range(int(np.ceil(stop / step))):
                batch = to_run[i * step : (i + 1) * step]
                self._call_executor(batch, **kwargs)

        if force_run_all:
            return self._quantum_results

        # Expand computed results to all results using counts.
        results_dict = dict(zip(collection.keys(), self._quantum_results))
        results = [results_dict[key] for key in hashable_circuits]

        return results

    def _call_executor(
        self, to_run: Union[QPROGRAM, Sequence[QPROGRAM]], **kwargs: Any
    ) -> None:
        """Calls the executor on the input circuit(s) to run. Stores the
        executed circuits in ``self._executed_circuits`` and the quantum
        results in ``self._quantum_results``.

        Args:
            to_run: Circuit(s) to _run.
        """
        result = self._executor(to_run, **kwargs)  # type: ignore
        self._calls_to_executor += 1

        if self.can_batch:
            self._quantum_results.extend(result)
            self._executed_circuits.extend(to_run)
        else:
            self._quantum_results.append(result)
            self._executed_circuits.append(to_run)

    @staticmethod
    def is_batched_executor(
        executor: Callable[[Union[QPROGRAM, Sequence[QPROGRAM]]], Any]
    ) -> bool:
        """Returns True if the input function is recognized as a "batched
        executor", else False.

        The executor is detected as "batched" if and only if it is annotated
        with a return type that is one of the following:

            * ``Iterable[QuantumResult]``
            * ``List[QuantumResult]``
            * ``Sequence[QuantumResult]``
            * ``Tuple[QuantumResult]``

        Otherwise, it is considered "serial".

        Batched executors can _run several quantum programs in a single call.
        See below.

        Args:
            executor: A "serial executor" (1) or a "batched executor" (2).

                (1) A function which inputs a single ``QPROGRAM`` and outputs a
                single ``QuantumResult``.
                (2) A function which inputs a list of ``QPROGRAM``s and outputs
                a list of ``QuantumResult``s (one for each ``QPROGRAM``).

        Returns:
            True if the executor is detected as batched, else False.
        """
        executor_annotation = inspect.getfullargspec(executor).annotations

        return executor_annotation.get("return") in (
            BatchedType[T]  # type: ignore[index]
            for BatchedType in [Iterable, List, Sequence, Tuple]
            for T in QuantumResult.__args__  # type: ignore[attr-defined]
        )
