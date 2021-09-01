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
"""Functions for creating mirror circuits
as defined in https://arxiv.org/abs/2008.11294 for
benchmarking quantum computers (with error mitigation)."""
from typing import Optional, List
from numpy import random
import networkx as nx
import cirq
from cirq.experiments.qubit_characterizations import _single_qubit_cliffords

single_q_cliffords = _single_qubit_cliffords()
cliffords = single_q_cliffords.c1_in_xy
paulis = [cirq.X, cirq.Y, cirq.Z, cirq.I]


def random_paulis(
    nqubits: int, random_state: random.RandomState
) -> cirq.Circuit:
    """Returns a circuit with randomly selected Pauli gates on each qubit.

    Args:
        nqubits: The number of qubits in the circuit.
        random_state: Random state to select Paulis I, X, Y, Z
    """
    return cirq.Circuit(
        paulis[random_state.randint(len(paulis))](cirq.LineQubit(x))
        for x in range(nqubits)
    )


def edge_grab(
    two_qubit_gate_prob: float,
    connectivity_graph: nx.Graph,
    random_state: random.RandomState,
) -> nx.Graph:
    """Returns a set of edges for which two qubit gates
    are to be applied given a two qubit gate density
    and the connectivity graph that must be satisfied.

    Args:
        two_qubit_gate_prob: Probability of an edge being chosen
            from the set of candidate edges.
        connectivity_graph: The connectivity graph for the backend
            on which the circuit will be run.
        random_state: Random state to select edges (uniformly at random).
    """
    connectivity_graph = connectivity_graph.copy()
    candidate_edges = nx.Graph()

    final_edges = nx.Graph()
    final_edges.add_nodes_from(connectivity_graph)

    while connectivity_graph.edges:
        num = random_state.randint(connectivity_graph.size())
        edges = list(connectivity_graph.edges)
        curr_edge = edges[num]
        candidate_edges.add_edge(*curr_edge)
        connectivity_graph.remove_nodes_from(curr_edge)

    for edge in candidate_edges.edges:
        if random_state.uniform(0.0, 1.0) < two_qubit_gate_prob:
            final_edges.add_edge(*edge)
    return final_edges


def random_cliffords(
    connectivity_graph: nx.Graph, random_state: random.RandomState
) -> cirq.Circuit:
    """Returns a circuit with a two-qubit Clifford gate applied
    to each edge in edges, and a random single-qubit
    Clifford gate applied to every other qubit.

    Args:
        connectivity_graph: A graph with the edges for which the
            two-qubit Clifford gate is to be applied.
        random_state: Random state to choose Cliffords (uniformly at random).
    """
    gates = [
        cirq.CNOT(cirq.LineQubit(a), cirq.LineQubit(b))
        for a, b in list(connectivity_graph.edges)
    ]
    qubits = nx.Graph()
    qubits.add_nodes_from(nx.isolates(connectivity_graph))
    gates.append(random_single_cliffords(qubits, random_state))
    return cirq.Circuit(gates)


def random_single_cliffords(
    connectivity_graph: nx.Graph, random_state: random.RandomState
) -> cirq.Circuit:
    """Returns a circuit with a random single-qubit Clifford gate
    applied on each given qubit.

    Args:
        connectivity_graph: A graph with each node representing a qubit for
            which a random single-qubit Clifford gate is to be applied.
        random_state: Random state to choose Cliffords (uniformly at random).
    """
    gates: List[cirq.Operation] = []
    for qubit in connectivity_graph.nodes:
        num = random_state.randint(len(cliffords))
        for clifford_gate in cliffords[num]:
            gates.append(clifford_gate(cirq.LineQubit(qubit)))
    return cirq.Circuit(gates)


def generate_mirror_circuit(
    nlayers: int,
    two_qubit_gate_prob: float,
    connectivity_graph: nx.Graph,
    seed: Optional[int] = None,
) -> cirq.Circuit:
    """Returns a randomized mirror circuit.

    Args:
        nlayers: The number of random Clifford layers to be generated.
        two_qubit_gate_prob: Probability of a two-qubit gate being applied.
        connectivity_graph: The connectivity graph of the backend
        on which the mirror circuit will be run. This is used
        to make sure 2-qubit gates are only run on connected qubits.
        seed: Seed for generating randomized mirror circuit.
    """
    if not 0 <= two_qubit_gate_prob <= 1:
        raise ValueError("two_qubit_gate_prob must be between 0 and 1")

    random_state = random.RandomState(seed)
    nqubits = connectivity_graph.number_of_nodes()
    single_qubit_cliffords = random_single_cliffords(
        connectivity_graph, random_state=random_state
    )

    forward_circuit = cirq.Circuit()

    quasi_inversion_circuit = cirq.Circuit()
    quasi_inverse_gates = []

    for _ in range(nlayers):
        forward_circuit.append(random_paulis(nqubits, random_state))

        selected_edges = edge_grab(
            two_qubit_gate_prob, connectivity_graph, random_state
        )
        circ = random_cliffords(selected_edges, random_state)
        forward_circuit.append(circ)

        quasi_inverse_gates.append(random_paulis(nqubits, random_state))
        quasi_inverse_gates.append(cirq.inverse(circ))

    quasi_inversion_circuit.append(
        gate for gate in reversed(quasi_inverse_gates)
    )

    rand_paulis = cirq.Circuit(random_paulis(nqubits, random_state))
    return (
        single_qubit_cliffords
        + forward_circuit
        + rand_paulis
        + quasi_inversion_circuit
        + cirq.inverse(single_qubit_cliffords)
        + cirq.measure(*cirq.LineQubit.range(nqubits))
    )
