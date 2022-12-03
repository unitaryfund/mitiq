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

from math import prod, sqrt
from typing import Any

from mitiq import Executor
from mitiq.calibration import Settings


class Calibration:
    """A calibration object which keeps track of, and aids in EM parameter tuning."""

    def __init__(
        self,
        executor: Executor,
        settings: Settings,
        ideal_executor: Executor | None = None,
    ):
        self.executor = executor
        self.ideal_executor = ideal_executor
        self.settings = settings
        self.circuits = self.settings.make_circuits()
        self.results: list[dict[str, Any]] = []

    def get_cost(self) -> dict[str, int]:
        """Return expected number of noisy expectation values required for
        calibration. If ideal_executor is defined the calibration process will
        simulate an equal number of expectation values."""
        num_circuits = len(self.circuits)
        num_methods = len(self.settings.mitigation_methods)
        num_options = prod(map(len, self.settings.method_params.values()))

        noisy = num_circuits * num_methods * num_options
        ideal = noisy if self.ideal_executor else 0
        return {
            "noisy_executions": noisy,
            "ideal_executions": ideal,
        }

    def run_circuits(self) -> None:
        """run the calibration circuits and store the ideal and noisy expectation values for each circuit."""
        expvals = []
        for circuit_data in self.circuits:
            circuit = circuit_data.circuit
            noisy_expval = self.executor(circuit)
            ideal_expval = (
                self.ideal_executor(circuit) if self.ideal_executor else 1.0
            )
            mitigated = {
                method: {"results": [], "method_improvement_factor": None}
                for method in self.settings.mitigation_methods
            }
            for (
                method,
                factory,
                scale_factors,
                scale_method,
                em_func,
            ) in self.settings.mitigate_functions():
                em_method = mitigated[method]["results"]
                em_method.append(
                    {
                        "extrapolation_method": factory.__name__,
                        "scale_factors": scale_factors,
                        "scale_noise_method": scale_method.__name__,
                        "mitigated_value": em_func(circuit, self.executor),
                    }
                )
            expvals.append(
                {
                    "unmitigated": noisy_expval,
                    "mitigated": mitigated,
                    "ideal": ideal_expval,
                }
            )
        self.results = expvals

    def compute_improvements(self):
        """compute improvement factors for each calibration result"""
        best_val = 0.0
        best_key = ""
        circuit_index = 0
        for i, result in enumerate(self.results):
            ideal = result["ideal"]
            unmitigated = result["unmitigated"]
            unmitigated_error = abs((ideal - unmitigated) / ideal)
            for method, di in result["mitigated"].items():
                results = di["results"]
                mitigated_vals = list(
                    map(lambda di: di["mitigated_value"], results)
                )
                method_improvement_factor = abs(unmitigated - ideal) / sqrt(
                    len(mitigated_vals)
                    * sum(
                        (mitigated_val - ideal) ** 2
                        for mitigated_val in mitigated_vals
                    )
                )
                di["method_improvement_factor"] = method_improvement_factor

                for res in di["results"]:
                    mitigated_expval = res["mitigated_value"]
                    diff = abs((ideal - mitigated_expval) / ideal)
                    error_diff = abs(unmitigated_error - diff)
                    if error_diff > best_val:
                        best_val = error_diff
                        best_key = method
                        circuit_index = i
        print("circuit index:", circuit_index)
        print("|ideal - best| / ideal =", best_val)
        return best_key

    def get_optimal_strategy(self):
        """uses the improvement factors to propose optimal error mitigation strategy"""
        pass

    def run(self):
        """make_circuits -> run_circuits -> compute_improvements"""
        pass
