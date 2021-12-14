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

"""Mitiq accuracy and timing benchmarks."""

import functools

import networkx as nx
import numpy as np

from mitiq import benchmarks, raw, zne, Observable, PauliString
from mitiq.interface import mitiq_cirq


compute_density_matrix_noiseless = functools.partial(
    mitiq_cirq.compute_density_matrix, noise_level=(0.0,)
)
benchmark_circuit_types = ("rb", "mirror")


def get_benchmark_circuit(
    circuit_type: str, nqubits: int, depth: int,
) -> "cirq.Circuit":
    """Returns a benchmark circuit.

    Args:
        circuit_type: Type of benchmark circuit.
        nqubits: Number of qubits.
        depth: Some proxy of depth for the circuit.
    """
    if circuit_type not in benchmark_circuit_types:
        raise ValueError(
            f"Unknown circuit type. Known types are {benchmark_circuit_types}."
        )
    if circuit_type == "rb":
        (circuit,) = benchmarks.generate_rb_circuits(
            n_qubits=nqubits, num_cliffords=depth, trials=1
        )
    elif circuit_type == "mirror":
        circuit, _ = benchmarks.generate_mirror_circuit(
            nlayers=depth,
            two_qubit_gate_prob=1.0,
            connectivity_graph=nx.complete_graph(nqubits),
        )
    return circuit


def track_zne(
    circuit_type: str, nqubits: int, depth: int, observable: Observable,
) -> float:
    """Returns the ZNE error mitigation factor, i.e., the ratio

    (error without ZNE) / (error with ZNE).

    Args:
        circuit_type: Type of benchmark circuit.
        nqubits: Number of qubits in the benchmark circuit.
        depth: Some proxy of depth in the benchmark circuit.
        observable: Observable to compute the expectation value of.
    """
    circuit = get_benchmark_circuit(circuit_type, nqubits, depth)
    true_value = raw.execute(
        circuit, compute_density_matrix_noiseless, observable
    )
    raw_value = raw.execute(
        circuit, mitiq_cirq.compute_density_matrix, observable
    )
    zne_value = zne.execute_with_zne(
        circuit, mitiq_cirq.compute_density_matrix, observable,
    )
    return np.real(abs(true_value - raw_value) / abs(true_value - zne_value))


track_zne.param_names = [
    "circuit",
    "nqubits",
    "depth",
    "observable",
]
track_zne.params = (
    benchmark_circuit_types,
    [2],
    [1, 2, 3, 4, 5],
    [Observable(PauliString("ZZ"))],
)
track_zne.unit = "Error mitigation factor"
