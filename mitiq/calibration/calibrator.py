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

from math import sqrt
from typing import Any, Callable, Dict, List, Optional, Union, cast
from collections import defaultdict
from dataclasses import dataclass

import cirq

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
    DefaultStrategy,
)


@dataclass
class Result:
    strategy: Strategy
    problem: BenchmarkProblem
    results: Dict[str, float]
    id: int = -1


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
        self.circuits = settings.make_circuits()
        self.results: List[Result] = []

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

    def run_circuits(self) -> List[Result]:
        """Runs all the circuits required for calibration.

        Returns:
            A collection of experimental results along with a summary of each
            :class:`BenchmarkProblem` that was run.
        """
        results = []

        strategies = self.settings.make_strategies()
        for problem in self.circuits:
            circuit = problem.circuit
            circuit.append(cirq.measure(circuit.all_qubits()))

            bitstring_to_measure = problem.most_likely_bitstring()
            expval_executor = convert_to_expval_executor(
                self.executor, bitstring_to_measure
            )

            noisy_value = expval_executor.evaluate(circuit)[0]

            for strategy in strategies:
                mitigated_value = strategy.mitigation_function(
                    circuit, expval_executor
                )

                result = {
                    "mitigated": mitigated_value,
                    "noisy": noisy_value,
                    "ideal": problem.largest_probability(),
                }
                summary = Result(
                    strategy=strategy, problem=problem, results=result
                )
                results.append(summary)

        return results

    def filter_problems(
        self, results: List[Result]
    ) -> Dict[int, List[Dict[str, float]]]:

        di = defaultdict(list)
        for res in results:
            di[res.strategy.id].append(res.results)

        return di

    def get_strategy(self, strategy_id: int) -> Strategy:
        for res in self.results:
            if res.strategy.id == strategy_id:
                return res.strategy
        return DefaultStrategy

    def compute_errors(self, results: List[Result]) -> Dict[int, float]:
        errors = {}
        strategy_results = self.filter_problems(results)
        for strategy_id, expectation_values in strategy_results.items():
            average_error = 0.0
            for result in expectation_values:
                average_error += abs(result["ideal"] - result["mitigated"])

            errors[strategy_id] = average_error / len(expectation_values)

        return errors

    def best_strategy(self, results: List[Result]) -> Strategy:
        """Finds the best strategy by using the parameters that had the
        smallest error.

        Args:
            results: Calibration experiment results. Obtained by first running
                :func:`run_circuits`.

        Returns:
            A single :class:`Strategy` object specifying the technique and
            parameters that performed best.
        """
        errors = self.compute_errors(results)

        best_strategy_id = min(errors, key=errors.__getitem__)
        return self.get_strategy(best_strategy_id)

    def run(self) -> None:
        results = self.run_circuits()
        self.results = results


def convert_to_expval_executor(
    executor: Executor,
    bitstring: str,
) -> Executor:
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
        return distribution.get(bitstring, 0)

    return Executor(expval_executor)


def execute_with_mitigation(
    circuit: QPROGRAM,
    executor: Union[Executor, Callable[[QPROGRAM], QuantumResult]],
    observable: Optional[Observable] = None,
    *,
    calibrator: Calibrator,
) -> QuantumResult:
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

    if not calibrator.results:
        calibrator.run()
    strategy = calibrator.best_strategy(calibrator.results)
    em_func = strategy.mitigation_function
    return em_func(circuit, executor=executor, observable=observable)
