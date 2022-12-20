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

from dataclasses import dataclass, astuple
from functools import partial
from itertools import product
from typing import Any, Callable, cast, Iterator, List, Dict, Tuple

import networkx as nx
import cirq

from mitiq._typing import QuantumResult
from mitiq.benchmarks import (
    generate_ghz_circuit,
    generate_mirror_circuit,
    generate_quantum_volume_circuit,
    generate_rb_circuits,
)

# from mitiq.pec import execute_with_pec
from mitiq.zne import execute_with_zne
from mitiq.zne.inference import LinearFactory, RichardsonFactory
from mitiq.zne.scaling import (
    fold_gates_at_random,
    fold_global,
)


@dataclass
class BenchmarkProblem:
    circuit: cirq.Circuit
    num_qubits: int
    circuit_depth: int
    type: str
    two_qubit_gate_count: int
    ideal_distribution: Dict[str, float]

    def __getitem__(self, keys: Tuple[str, ...]) -> Iterator[Any]:
        return iter(getattr(self, k) for k in keys)


@dataclass
class Strategy:
    technique: str
    technique_params: Dict[str, Any]
    mitigation_function: Callable[..., QuantumResult]

    def __iter__(self) -> Iterator[Any]:
        return iter(astuple(self))


class Settings:
    """A class to store settings relating to error mitigation calibration."""

    def __init__(
        self,
        techniques: List[str],
        circuit_types: List[str],
        num_qubits: int,
        circuit_depth: int,
        technique_params: Dict[str, Any],
    ):
        self.techniques = techniques
        self.technique_params = technique_params
        self.circuit_types = circuit_types
        self.num_qubits = num_qubits
        self.circuit_depth = circuit_depth

    def make_circuits(self) -> List[BenchmarkProblem]:
        """Generate the circuits to run in a calibration experiment via the
        parameters passed in initialization."""
        circuits = []
        nqubits, depth = self.num_qubits, self.circuit_depth
        for circuit_type in self.circuit_types:
            if circuit_type == "ghz":
                circuit = generate_ghz_circuit(nqubits)
                ideal = {"0" * nqubits: 0.5, "1" * nqubits: 0.5}
            elif circuit_type == "rb":
                circuit = generate_rb_circuits(nqubits, depth)[0]
                ideal = {"0" * nqubits: 1.0}
            elif circuit_type == "mirror":
                circuit, bitstring_list = generate_mirror_circuit(
                    nlayers=depth,
                    two_qubit_gate_prob=1.0,
                    connectivity_graph=nx.complete_graph(nqubits),
                )
                ideal_bitstring = "".join(map(str, bitstring_list))
                ideal = {ideal_bitstring: 1.0}
            elif circuit_type == "qv":
                circuit, _ = generate_quantum_volume_circuit(nqubits, depth)
                raise NotImplementedError(
                    "quantum volume circuits not yet supported in calibration"
                )

            else:
                raise ValueError(
                    "invalid value passed for `circuit_types`. Must be "
                    "one of `ghz`, `rb`, `mirror`, or `qv`, "
                    f"but got {circuit_type}."
                )

            circuit = cast(cirq.Circuit, circuit)
            two_qubit_gate_count = sum(
                [len(op.qubits) > 1 for op in circuit.all_operations()]
            )
            circuits.append(
                BenchmarkProblem(
                    circuit,
                    num_qubits=len(circuit.all_qubits()),
                    circuit_depth=len(circuit),
                    type=circuit_type,
                    two_qubit_gate_count=two_qubit_gate_count,
                    ideal_distribution=ideal,
                )
            )
        return circuits

    def make_strategies(self) -> List[Strategy]:
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
                        Strategy(
                            method,
                            {
                                "factory": factory.__name__,
                                "scale_factors": scale_factors,
                                "scale_method": scale_method.__name__,
                            },
                            em_func,
                        )
                    )
            # elif method == "pec":
            #     funcs.append(("pec", execute_with_pec))
            else:
                raise ValueError(
                    "Invalid value passed for mitigation_methods. "
                    "Must be one of `zne`, `pec`"
                )
        return funcs


ZNESettings = Settings(
    ["zne"],
    circuit_types=["ghz", "rb", "mirror"],
    num_qubits=2,
    circuit_depth=5,
    technique_params={
        "scale_factors": [[1.0, 2.0, 3.0], [1.0, 3.0, 5.0]],
        "scale_methods": [fold_global, fold_gates_at_random],
        "factories": [RichardsonFactory, LinearFactory],
    },
)
