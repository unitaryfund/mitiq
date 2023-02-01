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

from copy import deepcopy
import warnings
from collections import Counter
from math import sqrt
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
    cast,
    Sequence,
)

import cirq

from mitiq import (
    QPROGRAM,
    Executor,
    MeasurementResult,
    Observable,
    QuantumResult,
)
from mitiq.calibration.settings import Settings, Strategy, DefaultStrategy


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
        self.results: List[Dict[str, Any]] = []

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

    def run_circuits(self) -> List[Dict[str, Any]]:
        """Runs all the circuits required for calibration.

        Returns:
            A collection of experimental results along with a summary of each
            :class:`BenchmarkProblem` that was run.
        """
        expvals = []
        for problem in self.circuits:
            circuit, distribution = problem["circuit", "ideal_distribution"]
            circuit.append(cirq.measure(circuit.all_qubits()))

            expval_executor, bitstring_to_measure = convert_to_expval_executor(
                self.executor, distribution
            )
            noisy_value = expval_executor.evaluate(circuit)[0]
            ideal_value = distribution[bitstring_to_measure]
            noisy_error = ideal_value - noisy_value

            mitigated_values: Dict[str, Dict[str, Any]] = {
                technique.name: {"results": [], "improvement_factor": None}
                for technique in self.settings.techniques
            }
            for strategy in self.settings.make_strategies():
                mitigated_value = strategy.mitigation_function(
                    circuit, expval_executor
                )
                improvement_factor = abs(
                    noisy_error / (ideal_value - mitigated_value)
                )

                result = {
                    "mitigated_value": mitigated_value,
                    "improvement_factor": improvement_factor,
                    "strategy": strategy,
                    **strategy.as_dict(),
                }
                technique = strategy.technique.name
                mitigated_values[technique]["results"].append(result)

            circuit_info = problem.problem_summary_dict()
            expvals.append(
                {
                    "circuit_info": circuit_info,
                    "noisy_value": noisy_value,
                    "ideal_value": distribution[bitstring_to_measure],
                    "mitigated_values": mitigated_values,
                }
            )
        return expvals

    def compute_improvements(
        self, experiment_results: List[Dict[str, Any]]
    ) -> None:
        """Computes the improvement factors for each calibration circuit that
        was run. Saves the improvement factors in the input dictionary.

        Args:
            experiment_results: Results obtained from :func:`run_circuits`.
        """
        regularizing_epsilon = 1e-30
        for result in experiment_results:
            ideal_value = result["ideal_value"]
            noisy_value = result["noisy_value"]
            for di in result["mitigated_values"].values():
                results = di["results"]
                mitigated_values = [di["mitigated_value"] for di in results]
                improvement_factor = abs(noisy_value - ideal_value) / sqrt(
                    regularizing_epsilon
                    + len(mitigated_values)
                    * sum(
                        (mitigated_value - ideal_value) ** 2
                        for mitigated_value in mitigated_values
                    )
                )
                di["improvement_factor"] = improvement_factor

    def best_strategy(self, results: List[Dict[str, Any]]) -> Strategy:
        """Finds the best strategy by using the parameters that had the
        largest improvement factor.

        Args:
            results: Calibration experiment results. Obtained by first running
                :func:`run_circuits` and :func:`compute_improvements`.

        Returns:
            A single :class:`Strategy` object specifying the technique and
            parameters that performed best.
        """
        best_improvement_factor = 1.0
        num_circuits = len(self.settings.circuit_types)

        strategy = DefaultStrategy

        def filter_on_strategy(
            result: Dict[str, Dict[str, Dict[str, Any]]], strategy_id: int
        ) -> Dict[str, Dict[str, Dict[str, Any]]]:
            """Obtain results corresponding to the strategy of interest.
            Args:
                result: Calibration experiment results.
                strategy_id: Index of the strategy of interest.

            Returns:
                A dictionary of results corresponding to the strategy of
                interest.
            """
            res = result
            res["mitigated_values"]["ZNE"]["results"] = [
                res["mitigated_values"]["ZNE"]["results"][strategy_id]
            ]
            return res

        for strategy_id in range(
            len(results[0]["mitigated_values"]["ZNE"]["results"])
        ):
            strategy_group = []
            for c in range(num_circuits):
                result_copy = deepcopy(results)
                strategy_group.append(
                    filter_on_strategy(result_copy[c], strategy_id=strategy_id)
                )
                self.compute_improvements(strategy_group)

            if (
                strategy_group[0]["mitigated_values"]["ZNE"]["results"][0][
                    "improvement_factor"
                ]
                > best_improvement_factor
            ):
                best_improvement_factor = strategy_group[0][
                    "mitigated_values"
                ]["ZNE"]["results"][0]["improvement_factor"]
                strategy = strategy_group[0]["mitigated_values"]["ZNE"][
                    "results"
                ][0]["strategy"]
        self.results.append(
            {"best_improvement_factor": best_improvement_factor}
        )

        if strategy is DefaultStrategy:
            warnings.warn("None of the improvement factors were > 1")

        return strategy

    def run(self) -> None:
        results = self.run_circuits()
        self.compute_improvements(results)
        self.results = results


def bitstrings_to_distribution(
    bitstrings: Sequence[List[int]],
) -> Dict[str, float]:
    """Helper function to convert raw measurement results to probability
    distributions."""
    distribution = Counter(
        ["".join(map(str, bitstring)) for bitstring in bitstrings]
    )
    bitstring_count = len(bitstrings)
    for bitstring in distribution:
        distribution[bitstring] /= bitstring_count  # type: ignore [assignment]
    return dict(distribution)


def convert_to_expval_executor(
    executor: Executor,
    distribution: Optional[Dict[str, float]] = None,
    bitstring: Optional[str] = None,
) -> Tuple[Executor, str]:
    """Constructs a new executor returning an expectation value given by the
    probability that the circuit outputs the most likely state according to the
    ideal distribution.

    Args:
        executor: Executor which returns a :class:`.MeasurementResult`
            (bitstrings).
        distribution: The ideal distribution at the end of the circuit run.
        bitstring: The bitstring to measure the probability of.

    Returns:
        A tuple containing an executor returning expectation values and,
        the most likely bitstring, according to the passed ``distribution``
    """
    bitstring_to_measure = ""
    if distribution:
        bitstring_to_measure = max(
            distribution,
            key=distribution.get,  # type: ignore [arg-type]
        )
    elif bitstring:
        bitstring_to_measure = bitstring

    def expval_executor(circuit: cirq.Circuit) -> float:
        raw = cast(MeasurementResult, executor.run([circuit])[0]).result
        raw = cast(List[List[int]], raw)
        bitstring_distribution = bitstrings_to_distribution(raw)
        return bitstring_distribution.get(bitstring_to_measure, 0)

    return (
        Executor(expval_executor),  # type: ignore [arg-type]
        bitstring_to_measure,
    )


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
