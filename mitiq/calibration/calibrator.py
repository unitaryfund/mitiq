# Copyright (C) Unitary Fund
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

import warnings
from operator import itemgetter
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union, cast

import cirq
import numpy as np
import numpy.typing as npt
from tabulate import tabulate

from mitiq import (
    QPROGRAM,
    Executor,
    MeasurementResult,
    Observable,
    QuantumResult,
)
from mitiq.calibration.settings import (
    BenchmarkProblem,
    Settings,
    Strategy,
    ZNESettings,
)
from mitiq.interface import convert_from_mitiq


class MissingResultsError(Exception):
    pass


class ExperimentResults:
    """Class to store calibration experiment data, and provide helper methods
    for computing results based on it."""

    def __init__(
        self, strategies: List[Strategy], problems: List[BenchmarkProblem]
    ) -> None:
        self.strategies = strategies
        self.problems = problems
        self.num_strategies = len(strategies)
        self.num_problems = len(problems)
        self.reset_data()

    def add_result(
        self,
        strategy: Strategy,
        problem: BenchmarkProblem,
        *,
        ideal_val: float,
        noisy_val: float,
        mitigated_val: float,
    ) -> None:
        """Add a single result from a (Strategy, BenchmarkProblem) pair and
        store the results."""
        self.mitigated[strategy.id, problem.id] = mitigated_val
        self.noisy[strategy.id, problem.id] = noisy_val
        self.ideal[strategy.id, problem.id] = ideal_val

    def _get_performance(
        self, strategy_id: int, problem_id: int
    ) -> Tuple[str, float, float]:
        """Get performance symbol and errors.

        Returns:
            A performance tuple comprising:
            - performance symbol either ✔ or ✘ depending
              on whether mitigation technique works well or not.
              It considered to work well if the mitigated error is less
              than the noisy error.
            - the absolute value of the noisy error
            - the absolute value of the mitigated error
        """
        mitigated = self.mitigated[strategy_id, problem_id]
        noisy = self.noisy[strategy_id, problem_id]
        ideal = self.ideal[strategy_id, problem_id]
        mitigated_error = abs(ideal - mitigated)
        noisy_error = abs(ideal - noisy)
        mitigation_worked = mitigated_error < noisy_error
        performance_symbol = "✔" if mitigation_worked else "✘"
        return performance_symbol, noisy_error, mitigated_error

    def log_results(self) -> None:
        """Prints calibration results in the following form
        ┌────────────────────────────────────────────┬────────────────────────────────┬─────────────────────────┐
        │ benchmark                                  │ strategy                       │ performance             │
        ├────────────────────────────────────────────┼────────────────────────────────┼─────────────────────────┤
        │ Type: rb                                   │ Technique: ZNE                 │ ✔                       │
        │ Ideal distribution: {'00': 1.0}            │ Factory: RichardsonFactory     │ Noisy error: 0.1053     │
        │ Num qubits: 2                              │ Scale factors: [1.0, 2.0, 3.0] │ Mitigated error: 0.0146 │
        │ Circuit depth: 326                         │ Scale method: fold_global      │                         │
        │ Two qubit gate count: 79                   │                                │                         │
        ├────────────────────────────────────────────┼────────────────────────────────┼─────────────────────────┤
        │ Type: rb                                   │ Technique: ZNE                 │ ✔                       │
        │ Ideal distribution: {'00': 1.0}            │ Factory: RichardsonFactory     │ Noisy error: 0.1053     │
        │ Num qubits: 2                              │ Scale factors: [1.0, 3.0, 5.0] │ Mitigated error: 0.0422 │
        │ Circuit depth: 326                         │ Scale method: fold_global      │                         │
        │ Two qubit gate count: 79                   │                                │                         │
        ├────────────────────────────────────────────┼────────────────────────────────┼─────────────────────────┤
        │ Type: ghz                                  │ Technique: ZNE                 │ ✔                       │
        │ Ideal distribution: {'00': 0.5, '11': 0.5} │ Factory: RichardsonFactory     │ Noisy error: 0.0157     │
        │ Num qubits: 2                              │ Scale factors: [1.0, 2.0, 3.0] │ Mitigated error: 0.0018 │
        │ Circuit depth: 2                           │ Scale method: fold_global      │                         │
        │ Two qubit gate count: 1                    │                                │                         │
        ├────────────────────────────────────────────┼────────────────────────────────┼─────────────────────────┤
        │ Type: ghz                                  │ Technique: ZNE                 │ ✔                       │
        │ Ideal distribution: {'00': 0.5, '11': 0.5} │ Factory: RichardsonFactory     │ Noisy error: 0.0157     │
        │ Num qubits: 2                              │ Scale factors: [1.0, 3.0, 5.0] │ Mitigated error: 0.0091 │
        │ Circuit depth: 2                           │ Scale method: fold_global      │                         │
        │ Two qubit gate count: 1                    │                                │                         │
        └────────────────────────────────────────────┴────────────────────────────────┴─────────────────────────┘
        """  # noqa: E501
        table: List[List[str | float]] = []
        headers: List[str] = ["benchmark", "strategy", "performance"]
        for problem in self.problems:
            row_group: List[List[str | float]] = []
            for strategy in self.strategies:
                perf, nerr, merr = self._get_performance(
                    strategy.id, problem.id
                )
                row_group.append(
                    [
                        str(problem),
                        str(strategy),
                        f"{perf}\nNoisy error: {round(nerr, 4)}\n"
                        f"Mitigated error: {round(merr, 4)}",
                        # this is only for sorting
                        # removed after sorting
                        merr - nerr,
                    ]
                )
            row_group.sort(key=itemgetter(-1))
            table.extend([r[:-1] for r in row_group])
        return print(tabulate(table, headers, tablefmt="simple_grid"))

    def log_results_cartesian(self) -> None:
        """Prints calibration results in the following form
        ┌────────────────────────────────┬───────────────────────────────────┬──────────────────────────────────────────────┐
        │ strategy\benchmark             │ Type: rb                          │ Type: ghz                                    │
        │                                │ Ideal distribution: {'00': 1.0}   │ Ideal distribution: {'00': 0.5, '11': 0.5}   │
        │                                │ Num qubits: 2                     │ Num qubits: 2                                │
        │                                │ Circuit depth: 326                │ Circuit depth: 2                             │
        │                                │ Two qubit gate count: 79          │ Two qubit gate count: 1                      │
        │                                │                                   │                                              │
        ├────────────────────────────────┼───────────────────────────────────┼──────────────────────────────────────────────┤
        │ Technique: ZNE                 │ ✔                                 │ ✔                                            │
        │ Factory: RichardsonFactory     │ Noisy error: 0.1053               │ Noisy error: 0.0157                          │
        │ Scale factors: [1.0, 2.0, 3.0] │ Mitigated error: 0.0146           │ Mitigated error: 0.0018                      │
        │ Scale method: fold_global      │                                   │                                              │
        ├────────────────────────────────┼───────────────────────────────────┼──────────────────────────────────────────────┤
        │ Technique: ZNE                 │ ✔                                 │ ✔                                            │
        │ Factory: RichardsonFactory     │ Noisy error: 0.1053               │ Noisy error: 0.0157                          │
        │ Scale factors: [1.0, 3.0, 5.0] │ Mitigated error: 0.0422           │ Mitigated error: 0.0091                      │
        │ Scale method: fold_global      │                                   │                                              │
        └────────────────────────────────┴───────────────────────────────────┴──────────────────────────────────────────────┘
        """  # noqa: E501
        table: List[List[str]] = []
        headers: List[str] = ["strategy\\benchmark"]
        for problem in self.problems:
            headers.append(str(problem))
        for strategy in self.strategies:
            row: List[str] = [str(strategy)]
            for problem in self.problems:
                perf, nerr, merr = self._get_performance(
                    strategy.id, problem.id
                )
                row.append(
                    f"{perf}\nNoisy error: {round(nerr, 4)}\n"
                    f"Mitigated error: {round(merr, 4)}",
                )
            table.append(row)
        return print(tabulate(table, headers, tablefmt="simple_grid"))

    def is_missing_data(self) -> bool:
        """Method to check if there is any missing data that was expected from
        the calibration experiments."""
        return np.isnan(self.mitigated + self.noisy + self.ideal).any()

    def ensure_full(self) -> None:
        """Check to ensure all expected data is collected. All mitigated, noisy
        and ideal values must be nonempty for this to pass and return True."""
        if self.is_missing_data():
            raise MissingResultsError(
                "There are missing results from the expected calibration "
                "experiments. Please try running the experiments again with "
                "the `run` function."
            )

    def squared_errors(self) -> npt.NDArray[np.float32]:
        """Returns an array of squared errors, one for each (strategy, problem)
        pair."""
        return (self.ideal - self.mitigated) ** 2

    def best_strategy_id(self) -> int:
        """Returns the stategy id that corresponds to the strategy that
        maintained the smallest error across all ``BenchmarkProblem``
        instances."""
        errors = self.squared_errors()
        strategy_errors = np.sum(errors, axis=1)
        strategy_id = int(np.argmin(strategy_errors))
        return strategy_id

    def reset_data(self) -> None:
        """Reset all experiment result data using NaN values."""
        self.mitigated = np.full(
            (self.num_strategies, self.num_problems), np.nan
        )
        self.noisy = np.full((self.num_strategies, self.num_problems), np.nan)
        self.ideal = np.full((self.num_strategies, self.num_problems), np.nan)


class Calibrator:
    """An object used to orchestrate experiments for calibrating optimal error
    mitigation strategies.

    Args:
        executor: An unmitigated executor returning a
            :class:`.MeasurementResult`.
        settings: A ``Settings`` object which specifies the type and amount of
            circuits/error mitigation methods to run.
        frontend: The executor frontend as a string. For a list of supported
            frontends see ``mitiq.SUPPORTED_PROGRAM_TYPES.keys()``,
        ideal_executor: An optional simulated executor returning the ideal
            :class:`.MeasurementResult` without noise.
    """

    def __init__(
        self,
        executor: Union[Executor, Callable[[QPROGRAM], QuantumResult]],
        *,
        frontend: str,
        settings: Settings = ZNESettings,
        ideal_executor: Union[
            Executor, Callable[[QPROGRAM], QuantumResult], None
        ] = None,
    ):
        self.executor = (
            executor if isinstance(executor, Executor) else Executor(executor)
        )
        self.ideal_executor = (
            Executor(ideal_executor)
            if ideal_executor and not isinstance(ideal_executor, Executor)
            else None
        )
        self.settings = settings
        self.problems = settings.make_problems()
        self.strategies = settings.make_strategies()
        self.results = ExperimentResults(
            strategies=self.strategies, problems=self.problems
        )

        # Build an executor of Cirq circuits
        def cirq_execute(
            circuits: Sequence[cirq.Circuit],
        ) -> Sequence[MeasurementResult]:
            q_programs = [convert_from_mitiq(c, frontend) for c in circuits]
            results = cast(
                Sequence[MeasurementResult], self.executor.run(q_programs)
            )
            return results

        self._cirq_executor = Executor(cirq_execute)  # type: ignore [arg-type]

    @property
    def cirq_executor(self) -> Executor:
        """Returns an executor which is able to run Cirq circuits
        by converting them and calling self.executor.

        Args:
            executor: Executor which takes as input QPROGRAM circuits.

        Returns:
            Executor which takes as input a Cirq circuits.
        """
        return self._cirq_executor

    def get_cost(self) -> Dict[str, int]:
        """Returns the expected number of noisy and ideal expectation values
        required for calibration.

        Returns:
            A summary of the number of circuits to be run.
        """
        num_circuits = len(self.problems)
        num_options = sum(
            strategy.num_circuits_required() for strategy in self.strategies
        )

        noisy = num_circuits * num_options
        ideal = 0  # TODO: ideal executor is currently unused
        return {
            "noisy_executions": noisy,
            "ideal_executions": ideal,
        }

    def run(self, log: bool = False, log_cartesian: bool = False) -> None:
        """Runs all the circuits required for calibration."""
        if not self.results.is_missing_data():
            self.results.reset_data()

        for problem in self.problems:
            # Benchmark circuits have no measurements, so we append them.
            circuit = problem.circuit.copy()
            circuit.append(cirq.measure(circuit.all_qubits()))

            bitstring_to_measure = problem.most_likely_bitstring()
            expval_executor = convert_to_expval_executor(
                self.cirq_executor, bitstring_to_measure
            )
            noisy_value = expval_executor.evaluate(circuit)[0]
            for strategy in self.strategies:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    mitigated_value = strategy.mitigation_function(
                        circuit, expval_executor
                    )
                self.results.add_result(
                    strategy,
                    problem,
                    ideal_val=problem.largest_probability(),
                    noisy_val=noisy_value,
                    mitigated_val=mitigated_value,
                )

        self.results.ensure_full()

        if log:
            self.results.log_results()

        if log_cartesian:
            self.results.log_results_cartesian()

    def best_strategy(self) -> Strategy:
        """Finds the best strategy by using the parameters that had the
        smallest error.

        Args:
            results: Calibration experiment results. Obtained by first running
                :func:`run`.

        Returns:
            A single :class:`Strategy` object specifying the technique and
            parameters that performed best.
        """
        self.results.ensure_full()

        strategy_id = self.results.best_strategy_id()
        return self.settings.get_strategy(strategy_id)

    def execute_with_mitigation(
        self,
        circuit: QPROGRAM,
        expval_executor: Union[Executor, Callable[[QPROGRAM], QuantumResult]],
        observable: Optional[Observable] = None,
    ) -> Union[QuantumResult, None]:
        """See :func:`execute_with_mitigation` for signature and details."""
        return execute_with_mitigation(
            circuit, expval_executor, observable, calibrator=self
        )


def convert_to_expval_executor(executor: Executor, bitstring: str) -> Executor:
    """Constructs a new executor returning an expectation value given by the
    probability that the circuit outputs the most likely state according to the
    ideal distribution.

    Args:
        executor: Executor which returns a :class:`.MeasurementResult`
            (bitstrings).
        bitstring: The bitstring to measure the probability of. Defaults to
            ground state bitstring "00...0".

    Returns:
        A tuple containing an executor returning expectation values and,
        the most likely bitstring, according to the passed ``distribution``
    """

    def expval_executor(circuit: cirq.Circuit) -> float:
        circuit_with_meas = circuit.copy()
        if not cirq.is_measurement(circuit_with_meas):
            circuit_with_meas.append(
                cirq.measure(circuit_with_meas.all_qubits())
            )
        raw = cast(MeasurementResult, executor.run([circuit_with_meas])[0])
        distribution = raw.prob_distribution()
        return distribution.get(bitstring, 0.0)

    return Executor(expval_executor)  # type: ignore [arg-type]


def execute_with_mitigation(
    circuit: QPROGRAM,
    executor: Union[Executor, Callable[[QPROGRAM], QuantumResult]],
    observable: Optional[Observable] = None,
    *,
    calibrator: Calibrator,
) -> Union[QuantumResult, None]:
    """Estimates the error-mitigated expectation value associated to the
    input circuit, via the application of the best mitigation strategy, as
    determined by calibration.

    Args:
        circuit: The input circuit to execute.
        executor: A Mitiq executor that executes a circuit and returns the
            unmitigated ``QuantumResult`` (e.g. an expectation value).
        observable: Observable to compute the expectation value of. If
            ``None``, the ``executor`` must return an expectation value.
            Otherwise, the ``QuantumResult`` returned by ``executor`` is used
            to compute the expectation of the observable.
        calibrator: ``Calibrator`` object with which to determine the error
            mitigation strategy to execute the circuit.

    Returns:
        The error mitigated expectation expectation value.
    """

    if calibrator.results.is_missing_data():
        cost = calibrator.get_cost()
        answer = input(
            "Calibration experiments have not yet been run. You can run the "
            "experiments manually by calling `calibrator.run()`, or they can "
            f"be run now. The potential cost is:\n{cost}\n"
            "Would you like the experiments to be run automatically? (yes/no)"
        )
        if answer.lower() == "yes":
            calibrator.run()
        else:
            return None

    strategy = calibrator.best_strategy()
    em_func = strategy.mitigation_function
    return em_func(circuit, executor=executor, observable=observable)
