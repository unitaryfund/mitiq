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

from dataclasses import dataclass
from functools import partial
from itertools import product
from typing import Any, Callable

import networkx as nx

from mitiq._typing import QPROGRAM
from mitiq.benchmarks import (
    generate_ghz_circuit,
    generate_mirror_circuit,
    generate_quantum_volume_circuit,
    generate_rb_circuits,
)
from mitiq.pec import execute_with_pec
from mitiq.zne import execute_with_zne
from mitiq.zne.inference import ExpFactory, LinearFactory, RichardsonFactory
from mitiq.zne.scaling import (
    fold_gates_at_random,
    fold_global,
    insert_id_layers,
)


@dataclass
class CircuitData:
    circuit: QPROGRAM
    dimensions: tuple[int, int]
    type: str
    two_qubit_gate_count: int


class Settings:
    """A class to store settings relating to error mitigation calibration."""

    def __init__(
        self,
        techniques: list[str],
        circuit_types: list[str],
        circuit_dimensions: tuple[int, int],
        technique_params: dict[str, Any],
    ):
        self.techniques = techniques
        self.technique_params = technique_params
        self.circuit_types = circuit_types
        self.circuit_dimensions = circuit_dimensions

    def make_circuits(self) -> list[CircuitData]:
        """Generate the circuits to run in a calibration experiment via the
        parameters passed in initialization."""
        circuits = []
        nqubits, depth = self.circuit_dimensions
        for circuit_type in self.circuit_types:
            if circuit_type == "ghz":
                circuit = generate_ghz_circuit(nqubits)
            elif circuit_type == "rb":
                circuit = generate_rb_circuits(nqubits, depth)[0]
            elif circuit_type == "mirror":
                circuit = generate_mirror_circuit(
                    nlayers=depth,
                    two_qubit_gate_prob=1.0,
                    connectivity_graph=nx.complete_graph(nqubits),
                )[0]
            elif circuit_type == "qv":
                circuit = generate_quantum_volume_circuit(nqubits, depth)[0]
            else:
                raise ValueError(
                    f"invalid value passed for `circuit_types`. Must be one of `ghz`, `rb`, `mirror`, or `qv`, but got {circuit_type}"
                )

            circuit_dimensions = (len(circuit.all_qubits()), len(circuit))
            two_qubit_gate_count = sum(
                [len(op.qubits) > 1 for op in circuit.all_operations()]
            )
            circuits.append(
                CircuitData(
                    circuit,
                    circuit_dimensions,
                    circuit_type,
                    two_qubit_gate_count,
                )
            )
        return circuits

    def mitigate_functions(self) -> list[tuple[str, Callable]]:
        """Generates a list of ready to apply error mitigation functions
        preloaded with hyperparameters."""
        funcs = []
        for method in self.techniques:
            if method == "zne":
                for factory, scale_factors, scale_method in product(
                    self.technique_params["factories"],
                    self.technique_params["scale_factors"],
                    self.technique_params["scale_methods"],
                ):
                    inference_func = factory(scale_factors)
                    em_func = partial(
                        execute_with_zne,
                        factory=inference_func,
                        scale_noise=scale_method,
                    )
                    funcs.append(
                        ("zne", factory, scale_factors, scale_method, em_func)
                    )
            elif method == "pec":
                funcs.append(("pec", execute_with_pec))
            else:
                raise ValueError(
                    "invalid value passed for mitigation_methods. Must be one of `zne`, `pec`"
                )
        return funcs


ZNESettings = Settings(
    ["zne"],
    circuit_types=["ghz", "rb", "mirror"],
    circuit_dimensions=(2, 5),
    technique_params={
        "scale_factors": [[1.0, 2.0, 3.0], [1.0, 3.0, 5.0]],
        "scale_methods": [fold_global, fold_gates_at_random],
        "factories": [RichardsonFactory, LinearFactory],
    },
)
