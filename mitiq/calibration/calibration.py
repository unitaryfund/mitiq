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
from math import prod, sqrt
from typing import Any, Callable, Union, cast

import cirq

from mitiq import QPROGRAM, Executor, MeasurementResult, QuantumResult
from mitiq.calibration.settings import Settings


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
            Executor(executor)
            if not isinstance(executor, Executor)
            else executor
        )
        self.ideal_executor = (
            Executor(ideal_executor)
            if ideal_executor and not isinstance(ideal_executor, Executor)
            else ideal_executor
        )
        self.settings = settings
        self.circuits = self.settings.make_circuits()
        self.results: list[dict[str, Any]] = []

    def get_cost(self) -> dict[str, int]:
        """Returns the expected number of noisy expectation values required
        for calibration. If an ideal_executor was used in specifying the
        Calibrator object, the number of classical simulations is also
        returned."""
        num_circuits = len(self.circuits)
        num_methods = len(self.settings.techniques)
        num_options = prod(map(len, self.settings.technique_params.values()))

        noisy = num_circuits * num_methods * num_options
        ideal = noisy if self.ideal_executor else 0
        return {
            "noisy_executions": noisy,
            "ideal_executions": ideal,
        }

    def run_circuits(self) -> list[dict[str, Any]]:
        """Run the calibration circuits and store the ideal and noisy
        expectation values for each circuit in `self.results`."""
        expvals = []
        for circuit_data in self.circuits:
            circuit = circuit_data.circuit
            circuit.append(cirq.measure(circuit.all_qubits()))

            noisy_results = cast(
                MeasurementResult, self.executor._run([circuit])[0]
            ).result
            noisy_distribution = bitstrings_to_distribution(noisy_results)
            ideal_results = (
                cast(
                    MeasurementResult, self.ideal_executor._run([circuit])[0]
                ).result
                if self.ideal_executor
                else None
            )
            ideal_distribution = (
                bitstrings_to_distribution(ideal_results)
                if ideal_results
                else {"1" * len(circuit.all_qubits()): 1.0}
            )

            mitigated: dict[str, dict[str, Any]] = {
                technique: {"results": [], "improvement_factor": None}
                for technique in self.settings.techniques
            }
            bitstring_to_observe = max(
                ideal_distribution, key=ideal_distribution.get
            )
            for strategy in self.settings.make_strategies():
                expval_executor = bitstring_executor_to_expval_executor(
                    self.executor, bitstring_to_observe
                )
                mitigated_expval = strategy.mitigation_function(
                    circuit, expval_executor
                )
                mitigated[strategy.technique]["results"].append(
                    {
                        "circuit_type": circuit_data.type,
                        "mitigated_value": mitigated_expval,
                        **strategy.technique_params,
                    }
                )
            expvals.append(
                {
                    "noisy_value": noisy_distribution.get(
                        bitstring_to_observe, 0
                    ),
                    "ideal_value": ideal_distribution.get(
                        bitstring_to_observe, 0
                    ),
                    "mitigated": mitigated,
                }
            )
        return expvals

    def compute_improvements(
        self, experiment_results: list[dict[str, Any]]
    ) -> None:
        """Computes the improvement factors for each calibration circuit that
        was run. Saves the improvement factors in the input dictionary."""
        for result in experiment_results:
            ideal_value = result["ideal_value"]
            noisy_value = result["noisy_value"]
            for di in result["mitigated"].values():
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

    def get_optimal_strategy(self, results: list[dict[str, Any]]) -> str:
        """Finds the optimal error mitigation strategy using the improvement
        factors calculated, and stored in `self.results`.
        
        Currently, this function """
        best_val = 0.0
        best_key = ""
        for result in results:
            for method, di in result["mitigated"].items():
                improvement_factor = di["improvement_factor"]
                if improvement_factor > best_val:
                    best_val = improvement_factor
                    best_key = method
        return best_key

    def run(self) -> None:
        results = self.run_circuits()
        self.compute_improvements(results)
        self.results = results


def bitstrings_to_distribution(
    bitstrings: list[list[int]],
) -> dict[str, float]:
    """Helper function to convert raw measurement results to probability
    distributions."""
    distribution = Counter(
        ["".join(map(str, bitstring)) for bitstring in bitstrings]
    )
    bitstring_count = len(bitstrings)
    for bitstring in distribution:
        distribution[bitstring] /= bitstring_count
    return dict(distribution)


def bitstring_executor_to_expval_executor(
    ex: Executor, bitstring: str
) -> Executor:
    """Constructs a new executor returning expectation value given by the
    probability that the state is in state `bitstring`."""

    def expval_executor(circuit: cirq.Circuit) -> float:
        raw = cast(MeasurementResult, ex._run([circuit])[0]).result
        bitstring_distribution = bitstrings_to_distribution(raw)
        return bitstring_distribution.get(bitstring, 0) / len(raw)

    return Executor(expval_executor)
