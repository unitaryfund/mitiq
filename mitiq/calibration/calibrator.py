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

from typing import Callable, Dict, Optional, Union, cast
import warnings

import cirq
import numpy as np
import numpy.typing as npt

from mitiq import (
    QPROGRAM,
    Executor,
    MeasurementResult,
    Observable,
    QuantumResult,
)
from mitiq.calibration.settings import (
    Settings,
    Strategy,
    BenchmarkProblem,
)


class MissingResultsError(Exception):
    pass


class ExperimentResults:
    """Class to store calibration experiment data, and provide helper methods
    for computing results based on it."""

    def __init__(self, num_strategies: int, num_problems: int) -> None:
        self.num_strategies = num_strategies
        self.num_problems = num_problems
        self.reset_data()

    def add_result(
        self,
        strategy: Strategy,
        problem: BenchmarkProblem,
        *,
        ideal_val: float,
        noisy_val: float,
        mitigated_val: float,
        log: bool = False,
    ) -> None:
        """Add a single result from a (Strategy, BenchmarkProblem) pair and
        store the results."""
        self.mitigated[strategy.id, problem.id] = mitigated_val
        self.noisy[strategy.id, problem.id] = noisy_val
        self.ideal[strategy.id, problem.id] = ideal_val
        if not log:
            return
        mitigated_better = abs(ideal_val - mitigated_val) < abs(
            ideal_val - noisy_val
        )
        performance = "✅" if mitigated_better else "❌"
        print(
            f"Ran {problem.type} circuit using:",
            list(strategy.to_dict().values()),
        )
        print(
            f"{performance} ideal: {ideal_val:.2f}\t"
            f"noisy: {noisy_val:.2f}\t"
            f"mitigated: {mitigated_val:.2f}"
        )

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
        ideal_executor: An optional simulated executor returning the ideal
            :class:`.MeasurementResult` without noise.
    """

    def __init__(
        self,
        executor: Union[Executor, Callable[[QPROGRAM], QuantumResult]],
        settings: Settings,
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
        self.circuits = settings.make_problems()
        self.strategies = settings.make_strategies()
        self.results = ExperimentResults(
            num_strategies=len(self.strategies),
            num_problems=len(self.circuits),
        )

    def get_cost(self) -> Dict[str, int]:
        """Returns the expected number of noisy and ideal expectation values
        required for calibration.

        Returns:
            A summary of the number of circuits to be run.
        """
        num_circuits = len(self.circuits)
        num_options = len(self.settings.technique_params)

        noisy = num_circuits * num_options
        ideal = 0  # TODO: ideal executor is currently unused
        return {
            "noisy_executions": noisy,
            "ideal_executions": ideal,
        }

    def run(self, log: bool = False) -> None:
        """Runs all the circuits required for calibration."""
        if not self.results.is_missing_data():
            self.results.reset_data()

        for problem in self.circuits:
            circuit = problem.circuit.copy()
            circuit.append(cirq.measure(circuit.all_qubits()))

            bitstring_to_measure = problem.most_likely_bitstring()
            expval_executor = convert_to_expval_executor(
                self.executor, bitstring_to_measure
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
                    log=log,
                )
        self.results.ensure_full()

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


def convert_to_expval_executor(executor: Executor, bitstring: str) -> Executor:
    """Constructs a new executor returning an expectation value given by the
    probability that the circuit outputs the most likely state according to the
    ideal distribution.

    Args:
        executor: Executor which returns a :class:`.MeasurementResult`
            (bitstrings).
        bitstring: The bitstring to measure the probability of.

    Returns:
        A tuple containing an executor returning expectation values and,
        the most likely bitstring, according to the passed ``distribution``
    """

    def expval_executor(circuit: cirq.Circuit) -> float:
        raw = cast(MeasurementResult, executor.run([circuit])[0])
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
