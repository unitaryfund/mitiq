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
import warnings
import inspect
from typing import (
    Any,
    Callable,
    cast,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
import numpy.typing as npt

from mitiq import QPROGRAM, MeasurementResult, QuantumResult

from mitiq.observable.observable import Observable
from mitiq.interface import convert_from_mitiq, convert_to_mitiq
from mitiq.observable.pauli import PauliString


DensityMatrixLike = [
    np.ndarray,
    Iterable[np.ndarray],  # type: ignore
    List[np.ndarray],  # type: ignore
    Sequence[np.ndarray],  # type: ignore
    Tuple[np.ndarray],
    npt.NDArray[np.complex64],
]
FloatLike = [
    None,  # Untyped executors are assumed to return floats.
    float,
    Iterable[float],
    List[float],
    Sequence[float],
    Tuple[float],
]
MeasurementResultLike = [
    MeasurementResult,
    Iterable[MeasurementResult],
    List[MeasurementResult],
    Sequence[MeasurementResult],
    Tuple[MeasurementResult],
]


class Executor:
    """Tool for efficiently scheduling/executing quantum programs and storing
    the results.
    """

    def __init__(
        self,
        executor: Callable[[Union[QPROGRAM, Sequence[QPROGRAM]]], Any],
        max_batch_size: int = 75,
    ) -> None:
        """Initializes an Executor.

        Args:
            executor: A function which inputs a program and outputs a
                ``mitiq.QuantumResult``, or inputs a sequence of programs and
                outputs a sequence of ``mitiq.QuantumResult`` s.
            max_batch_size: Maximum number of programs that can be sent in a
                single batch (if the executor is batched).
        """
        self._executor = executor

        executor_annotation = inspect.getfullargspec(executor).annotations
        self._executor_return_type = executor_annotation.get("return")
        self._max_batch_size = max_batch_size

        self._executed_circuits: List[QPROGRAM] = []
        self._quantum_results: List[QuantumResult] = []

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
    ) -> List[float]:
        """Returns the expectation value Tr[ρ O] for each circuit in
        ``circuits`` where O is the observable provided or implicitly defined
        by the ``executor``. (The observable is implicitly defined when the
        ``executor`` returns float(s).)

        All executed circuits are stored in ``self.executed_circuits``, and all
        quantum results are stored in ``self.quantum_results``.

        Args:
             circuits: A single circuit of list of circuits.
             observable: Observable O in the expression Tr[ρ O]. If None,
                the ``executor`` must return a float (which corresponds to
                Tr[ρ O] for a specific, fixed observable O).
            force_run_all: If True, force every circuit in the input sequence
                to be executed (if some are identical). Else, detects identical
                circuits and runs a minimal set.

        Returns:
            List of real valued expectation values.
        """
        if not isinstance(circuits, List):
            circuits = [circuits]

        warn_non_hermitian = False
        if observable:
            if isinstance(observable, PauliString):
                if observable.coeff.imag > 0.0001:
                    warn_non_hermitian = True
            elif isinstance(observable, Observable):
                if any(
                    pauli.coeff.imag > 0.0001 for pauli in observable._paulis
                ):
                    warn_non_hermitian = True
        if warn_non_hermitian:
            warnings.warn(
                "Expected observable to be hermitian. Continue with caution."
            )

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
        elif (
            observable is not None
            and self._executor_return_type not in MeasurementResultLike
            and self._executor_return_type not in DensityMatrixLike
        ):
            raise ValueError(
                """Executor and observable are not compatible. Executors
                returning expectation values as float must be used with
                observable=None"""
            )
        else:
            all_circuits = circuits
            result_step = 1

        # Run all required circuits.
        all_results = self._run(all_circuits, force_run_all, **kwargs)

        # Parse the results.
        if self._executor_return_type in FloatLike:
            results = np.real_if_close(
                cast(Sequence[float], all_results)
            ).tolist()

        elif self._executor_return_type in DensityMatrixLike:
            observable = cast(Observable, observable)
            all_results = cast(List[npt.NDArray[np.complex64]], all_results)
            results = [
                observable._expectation_from_density_matrix(density_matrix)
                for density_matrix in all_results
            ]

        elif self._executor_return_type in MeasurementResultLike:
            observable = cast(Observable, observable)
            all_results = cast(List[MeasurementResult], all_results)
            results = [
                observable._expectation_from_measurements(
                    all_results[i : i + result_step]
                )
                for i in range(len(all_results) // result_step)
            ]

        else:
            raise ValueError(
                f"Could not parse executed results from executor with type"
                f" {self._executor_return_type}."
            )

        return results  # type: ignore[return-value]

    def _run(
        self,
        circuits: Sequence[QPROGRAM],
        force_run_all: bool = False,
        **kwargs: Any,
    ) -> Sequence[QuantumResult]:
        """Runs all input circuits using the least number of possible calls to
        the executor.

        Args:
            circuits: Sequence of circuits to execute using the executor.
            force_run_all: If True, force every circuit in the input sequence
            to be executed (if some are identical). Else, detects identical
            circuits and runs a minimal set.
        """
        start_result_index = len(self._quantum_results)

        if force_run_all:
            to_run = circuits
        else:
            # Make circuits hashable.
            # Note: Assumes all circuits are the same type.
            # TODO: Bug! These conversions to/from Mitiq are not safe in that,
            #  e.g., they do not preserve classical register structure in
            #  Qiskit circuits, potentially causing executed results to be
            #  incorrect. Safe conversions should follow the logic in
            #  mitiq.interface.noise_scaling_converter.
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

        these_results = self._quantum_results[start_result_index:]

        if force_run_all:
            return these_results

        # Expand computed results to all results using counts.
        results_dict = dict(zip(collection.keys(), these_results))
        results = [results_dict[key] for key in hashable_circuits]

        return results

    def _call_executor(
        self, to_run: Union[QPROGRAM, Sequence[QPROGRAM]], **kwargs: Any
    ) -> None:
        """Calls the executor on the input circuit(s) to run. Stores the
        executed circuits in ``self._executed_circuits`` and the quantum
        results in ``self._quantum_results``.

        Args:
            to_run: Circuit(s) to run.
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
