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

from collections import Counter
from dataclasses import asdict
from math import prod, sqrt
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, cast

import cirq

from mitiq import (
    QPROGRAM,
    Executor,
    MeasurementResult,
    Observable,
    QuantumResult,
)
from mitiq.calibration.settings import Settings, Strategy


class Calibrator:
    """A calibration object which keeps track of, and aids in Error
    Mitigation parameter tuning."""

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
        self.circuits = self.settings.make_circuits()
        self.results: List[Dict[str, Any]] = []

    def get_cost(self) -> Dict[str, int]:
        """Returns the expected number of noisy expectation values required
        for calibration. If an ideal_executor was used in specifying the
        Calibrator object, the number of classical simulations is also
        returned."""
        num_circuits = len(self.circuits)
        num_methods = len(self.settings.techniques)
        num_options = prod(map(len, self.settings.technique_params.values()))

        noisy = num_circuits * num_methods * num_options
        ideal = 0  # TODO: ideal executor is currently unused
        return {
            "noisy_executions": noisy,
            "ideal_executions": ideal,
        }

    def run_circuits(self) -> List[Dict[str, Any]]:
        """Run the calibration circuits and store the ideal and noisy
        expectation values for each circuit in `self.results`."""
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
                technique: {"results": [], "improvement_factor": None}
                for technique in self.settings.techniques
            }
            for strategy in self.settings.make_strategies():
                technique, params, mitiq_func = strategy
                mitigated_value = mitiq_func(circuit, expval_executor)

                improvement_factor = abs(
                    noisy_error / (ideal_value - mitigated_value)
                )
                result = {
                    "mitigated_value": mitigated_value,
                    "improvement_factor": improvement_factor,
                    "strategy": strategy,
                    **params,
                }
                mitigated_values[technique]["results"].append(result)

            circuit_info = asdict(problem)
            # remove circuit; it can be regenerated if needed
            del circuit_info["circuit"]
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
        was run. Saves the improvement factors in the input dictionary."""
        for result in experiment_results:
            ideal_value = result["ideal_value"]
            noisy_value = result["noisy_value"]
            for di in result["mitigated_values"].values():
                results = di["results"]
                mitigated_values = list(
                    map(lambda di: di["mitigated_value"], results)
                )
                improvement_factor = abs(noisy_value - ideal_value) / sqrt(
                    len(mitigated_values)
                    * sum(
                        (mitigated_value - ideal_value) ** 2
                        for mitigated_value in mitigated_values
                    )
                )
                di["improvement_factor"] = improvement_factor

    def best_strategy(self, results: List[Dict[str, Any]]) -> Strategy:
        """Finds the best strategy by using the parameters that had the
        largest improvement factor."""
        best_improvement_factor = 1.0
        strategy = None
        for circuit in results:
            for result in circuit["mitigated_values"]["zne"]["results"]:
                if result["improvement_factor"] > best_improvement_factor:
                    best_improvement_factor = result["improvement_factor"]
                    strategy = result["strategy"]
        if strategy is None:
            raise ValueError("None of the improvement factors were > 0")
        return strategy

    def run(self) -> None:
        results = self.run_circuits()
        self.compute_improvements(results)
        self.results = results


def bitstrings_to_distribution(
    bitstrings: List[List[int]],
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
    ex: Executor,
    distribution: Optional[Dict[str, float]] = None,
    bitstring: Optional[str] = None,
) -> Tuple[Executor, str]:
    """Constructs a new executor returning an expectation value given by the
    probability that the circuit outputs the most likely state according to the
    ideal distribution.

    Returns:
        A two-tuple containing:
            - An executor returning expectation values: QPROGRAM -> float
            - The most likely bitstring, according to the passed `distribution`
    """
    bitstring_to_measure = ""
    if distribution:
        bitstring_to_measure = max(
            distribution,
            key=distribution.get,  # type: ignore [arg-type]
        )
    elif bitstring:
        bitstring = bitstring

    def expval_executor(circuit: cirq.Circuit) -> float:
        raw = cast(MeasurementResult, ex._run([circuit])[0]).result
        bitstring_distribution = bitstrings_to_distribution(raw)
        return bitstring_distribution.get(bitstring_to_measure, 0)

    return (
        Executor(expval_executor),  # type: ignore [arg-type]
        bitstring_to_measure,
    )


def execute_with_mitigation(
    circuit: QPROGRAM,
    calibrator: Calibrator,
    bitstring: str,
    observable: Optional[Observable] = None,
) -> QuantumResult:
    if not calibrator.results:
        calibrator.run()
    strategy = calibrator.best_strategy(calibrator.results)
    em_func = strategy.mitigation_function
    expval_executor, _ = convert_to_expval_executor(
        calibrator.executor, bitstring=bitstring
    )
    return em_func(circuit, executor=expval_executor, observable=observable)
