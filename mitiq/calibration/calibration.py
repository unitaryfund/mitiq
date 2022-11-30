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

from mitiq import Executor
from mitiq.calibration import Settings


class Calibration:
    """A calibration object which keeps track of, and aids in EM parameter tuning."""

    def __init__(
        self,
        executor: Executor,
        ideal_executor: Executor,
        settings: Settings,
    ):
        self.executor = executor
        self.ideal_executor = ideal_executor
        self.settings = settings
        self.circuits = self.settings.make_circuits()
        self.results = []

    def get_cost(self) -> dict[str, int]:
        """Return expected number of noisy expectation values required for calibration.
        If method requires simulation, returns number of ideal expectation values needed."""
        num_circuits = len(self.circuits)
        num_methods = len(self.settings.mitigation_methods)

        noisy = num_circuits * self.settings.max_executions * num_methods
        ideal = noisy if self.ideal_executor else 0
        return {
            "noisy_executions": noisy,
            "ideal_executions": ideal,
        }

    def run_circuits(self) -> list[dict[str, float]]:
        """run the calibration circuits and store the ideal and noisy expectation values for each circuit."""
        expvals = []
        for circuit_data in self.circuits:
            circuit = circuit_data.circuit
            expval = self.executor(circuit)
            mitigated = {}
            for (
                method_key,
                factory,
                scale_factors,
                scale_method,
                mitiq_func,
            ) in self.settings.mitigate_functions():
                mitigated[
                    f"{method_key}-{factory}-{scale_factors}-{scale_method}"
                ] = mitiq_func(circuit, self.executor)
            ideal = self.ideal_executor(circuit)
            expvals.append(
                {"unmitigated": expval, "mitigated": mitigated, "ideal": ideal}
            )
        self.results = expvals
        # return expvals

    def compute_improvements(self):
        """compute improvement factors for each calibration result"""
        best_val = 0.0
        best_key = ""
        circuit_index = 0
        for i, result in enumerate(self.results):
            ideal = result["ideal"]
            unmitigated = result["unmitigated"]
            unmitigated_error = abs((ideal - unmitigated) / ideal)
            for em_key, expval in result["mitigated"].items():
                diff = abs((ideal - expval) / ideal)
                error_diff = abs(unmitigated_error - diff)
                if error_diff > best_val:
                    best_val = error_diff
                    best_key = em_key
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
