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

from dataclasses import dataclass, astuple, asdict
from functools import partial
from typing import Any, Callable, cast, Iterator, List, Dict, Tuple, Optional
from enum import Enum, auto

import networkx as nx
import cirq

from mitiq import QuantumResult
from mitiq.benchmarks import (
    generate_ghz_circuit,
    generate_mirror_circuit,
    generate_quantum_volume_circuit,
    generate_rb_circuits,
)

from mitiq.pec import execute_with_pec
from mitiq.raw import execute
from mitiq.zne import execute_with_zne
from mitiq.zne.inference import LinearFactory, RichardsonFactory
from mitiq.zne.scaling import (
    fold_gates_at_random,
    fold_global,
)


class MitigationTechnique(Enum):
    """Simple enum type for handling validation, and providing helper functions
    when accessing mitigation techniques."""

    ZNE = auto()
    PEC = auto()
    RAW = auto()

    @property
    def mitigation_function(self) -> Callable[..., QuantumResult]:
        if self is MitigationTechnique.ZNE:
            return execute_with_zne
        elif self is MitigationTechnique.PEC:
            return cast(Callable[..., float], execute_with_pec)
        elif self is MitigationTechnique.RAW:
            return execute


@dataclass
class BenchmarkProblem:
    """A dataclass containing information for instances of problems that will
    be run during the calibrations process.

    Args:
        circuit: The circuit to be run.
        type: The type of the circuit (often the name of the algorithm)
        ideal_distribution: The ideal probability distribution after applying
            ``circuit``.
    """

    circuit: cirq.Circuit
    type: str
    ideal_distribution: Dict[str, float]

    def __getitem__(self, keys: Tuple[str, ...]) -> Iterator[Any]:
        return iter(getattr(self, k) for k in keys)

    @property
    def num_qubits(self) -> int:
        return len(self.circuit.all_qubits())

    @property
    def circuit_depth(self) -> int:
        return len(self.circuit)

    @property
    def two_qubit_gate_count(self) -> int:
        return sum(
            [len(op.qubits) > 1 for op in self.circuit.all_operations()]
        )

    def problem_summary_dict(self) -> Dict[str, Any]:
        """Produces a summary of the ``BenchmarkProblem``, to be used in
        recording the results when running calibration experiments.

        Returns:
            Dictionary summarizing important attributes of the problem's
            circuit.
        """
        base = asdict(self)
        # remove circuit; it can be regenerated if needed
        del base["circuit"]
        base["num_qubits"] = self.num_qubits
        base["circuit_depth"] = self.circuit_depth
        base["two_qubit_gate_count"] = self.two_qubit_gate_count
        return base


@dataclass
class Strategy:
    """A dataclass which describes precisely an error mitigation approach by
    specifying a technique and the associated options.

    Args:
        technique: One of Mitiq's support error mitigation strategies,
            specified as a :class:`MitigationTechnique`.
        technique_params: A dictionary of options to pass to the mitigation
            method specified in `technique`.
    """

    technique: MitigationTechnique
    technique_params: Dict[str, Any]

    @property
    def mitigation_function(self) -> Callable[..., QuantumResult]:
        return partial(
            self.technique.mitigation_function, **self.technique_params
        )

    def as_dict(self) -> Dict[str, Any]:
        """A summary of the strategies parameters, without the technique added.

        Returns:
            A dictionary describing the strategies parameters."""
        di = {}
        if self.technique is MitigationTechnique.ZNE:
            inference_func = self.technique_params["factory"]
            di["factory"] = inference_func.__class__.__name__
            di["scale_factors"] = inference_func.get_scale_factors().tolist()
            di["scale_method"] = self.technique_params["scale_noise"].__name__
        return di

    def __iter__(self) -> Iterator[Any]:
        return iter(astuple(self))

    def __str__(self) -> str:
        di = self.as_dict()
        di["technique"] = self.technique.name
        return str(di)


class Settings:
    """A class to store the configuration settings of a :class:`.Calibrator`.

    Args:
        circuit_types: List of strings specifying circuit types to use.
            Must be drawn from the Identifier column::

                Identifier  | Circuit Type
                ----------------------------
                "ghz"       | GHZ circuit
                "rb"        | Randomized Benchmarking
                "mirror"    | Mirror circuit

        num_qubits: Number of qubits to use for circuit generation.
        circuit_depth: Circuit depth to use when generating circuits. Only
            used when ``num_qubits`` in combination with the circuit type does
            not specify the depth.
        strategies: A specification of the methods/parameters to be used in
            calibration experiments.
        seed: A random seed for (mirror) circuit generation.
    """

    def __init__(
        self,
        circuit_types: List[str],
        num_qubits: int,
        circuit_depth: int,
        strategies: List[Dict[str, Any]],
        circuit_seed: Optional[int] = None,
    ):
        self.techniques = [
            MitigationTechnique[technique["technique"].upper()]
            for technique in strategies
        ]
        self.technique_params = strategies
        self.circuit_types = circuit_types
        self.num_qubits = num_qubits
        self.circuit_depth = circuit_depth
        self.circuit_seed = circuit_seed

    def make_circuits(self) -> List[BenchmarkProblem]:
        """Generate the circuits to run for the calibration experiment.
        Returns:
            A list of :class:`BenchmarkProblem` objects"""
        circuits = []
        nqubits, depth, seed = (
            self.num_qubits,
            self.circuit_depth,
            self.circuit_seed,
        )
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
                    seed=seed,
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
            circuits.append(
                BenchmarkProblem(
                    circuit,
                    type=circuit_type,
                    ideal_distribution=ideal,
                )
            )
        return circuits

    def make_strategies(self) -> List[Strategy]:
        """Generates a list of :class:`Strategy` objects using the specified
        configurations.

        Returns:
            A list of :class:`Strategy` objects."""
        funcs = []
        for technique, params in zip(self.techniques, self.technique_params):
            params_copy = params.copy()
            del params_copy["technique"]
            funcs.append(
                Strategy(technique=technique, technique_params=params_copy)
            )
        return funcs


ZNESettings = Settings(
    circuit_types=["ghz", "rb", "mirror"],
    num_qubits=2,
    circuit_depth=5,
    strategies=[
        {
            "technique": "zne",
            "scale_noise": fold_global,
            "factory": RichardsonFactory([1.0, 2.0, 3.0]),
        },
        {
            "technique": "zne",
            "scale_noise": fold_global,
            "factory": RichardsonFactory([1.0, 3.0, 5.0]),
        },
        {
            "technique": "zne",
            "scale_noise": fold_global,
            "factory": LinearFactory([1.0, 2.0, 3.0]),
        },
        {
            "technique": "zne",
            "scale_noise": fold_global,
            "factory": LinearFactory([1.0, 3.0, 5.0]),
        },
        {
            "technique": "zne",
            "scale_noise": fold_gates_at_random,
            "factory": RichardsonFactory([1.0, 2.0, 3.0]),
        },
        {
            "technique": "zne",
            "scale_noise": fold_gates_at_random,
            "factory": RichardsonFactory([1.0, 3.0, 5.0]),
        },
        {
            "technique": "zne",
            "scale_noise": fold_gates_at_random,
            "factory": LinearFactory([1.0, 2.0, 3.0]),
        },
        {
            "technique": "zne",
            "scale_noise": fold_gates_at_random,
            "factory": LinearFactory([1.0, 3.0, 5.0]),
        },
    ],
)

DefaultStrategy = Strategy(MitigationTechnique.RAW, {})
