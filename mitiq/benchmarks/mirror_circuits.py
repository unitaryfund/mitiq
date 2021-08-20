#!/usr/bin/env python
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

from typing import Optional, List
from numpy import random
import networkx as nx
import cirq
from cirq.experiments.qubit_characterizations import _single_qubit_cliffords

single_q_cliffords = _single_qubit_cliffords()
cliffords = single_q_cliffords.c1_in_xy
paulis = [cirq.X, cirq.Y, cirq.Z, cirq.I]


def random_paulis(nqubits: int) -> cirq.Circuit:
    """Returns a circuit with a random pauli gate applied to each qubit.

    Args:
        nqubits: The number of qubits in the circuit.
    """
    return cirq.Circuit(
        paulis[random.randint(4)](cirq.LineQubit(x)) for x in range(nqubits)
    )


def edge_grab(
    two_qubit_gate_prob: float, connectivity_graph_: nx.Graph
) -> nx.Graph:
    """Returns a set of edges for which two qubit gates
    are to be applied given a two qubit gate density, xi,
    and the connectivity graph that must be satisfied.

    Args:
        two_qubit_gate_prob: Probability of an edge being chosen
            from the set of candidate edges.
            connectivity_graph_: The connectivity graph for the backend
            on which the circuit will be run.
    """
    connectivity_graph = connectivity_graph_.copy()
    candidate_edges = nx.Graph()

    while connectivity_graph.edges:
        num = random.randint(connectivity_graph.size())
        edges = list(connectivity_graph.edges)
        curr_edge = edges[num]
        candidate_edges.add_edge(*curr_edge)
        connectivity_graph.remove_nodes_from(curr_edge)

    prob = two_qubit_gate_prob
    final_edges = nx.Graph()
    final_edges.add_nodes_from(connectivity_graph_)
    for edge in candidate_edges.edges:
        if random.random() < prob:
            final_edges.add_edge(*edge)
    return final_edges


def random_cliffords(edges: nx.Graph) -> cirq.Circuit:
    """Returns a circuit with a two-qubit Clifford gate applied
    to each edge in edges, and a random single-qubit
    Clifford gate applied to every other qubit.

    Args:
        edges: A graph with the edges for which the
            two-qubit Clifford gate is to be applied.
    """
    gates = []
    gates.append(
        cirq.CNOT(cirq.LineQubit(a), cirq.LineQubit(b)) for a, b in edges.edges
    )
    circuit = cirq.Circuit(gates)
    qubits = nx.Graph()
    qubits.add_nodes_from(nx.isolates(edges))
    circuit.append(random_single_cliffords(qubits))
    return circuit


def random_single_cliffords(qubits: nx.Graph) -> cirq.Circuit:
    """Returns a circuit with a random single-qubit Clifford gate
    applied on each given qubit.

    Args:
        qubits: A graph with each node representing a qubit for
            which a random single-qubit Clifford gate is to be applied.

    """
    gates: List[cirq.Operation] = []
    for qubit in qubits.nodes:
        num = random.randint(len(cliffords))
        gates.extend(
            clifford.on(cirq.LineQubit(qubit)) for clifford in cliffords[num]
        )
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
    """
    if not 0 <= two_qubit_gate_prob <= 1:
        raise ValueError("two_qubit_gate_prob must be between 0 and 1")

    random.seed(seed)

    nqubits = connectivity_graph.number_of_nodes()
    single_qubit_cliffords = random_single_cliffords(connectivity_graph)

    forward_circuit = cirq.Circuit()

    quasi_inversion_circuit = cirq.Circuit()
    quasi_inverse_gates = []

    for _ in range(nlayers):
        forward_circuit.append(random_paulis(nqubits))

        selected_edges = edge_grab(two_qubit_gate_prob, connectivity_graph)
        edge_graph = nx.Graph()
        edge_graph.add_nodes_from(connectivity_graph)
        edge_graph.add_edges_from(selected_edges.edges)
        circ = random_cliffords(edge_graph)
        forward_circuit.append(circ)

        quasi_inverse_gates.append(random_paulis(nqubits))
        quasi_inverse_gates.append(cirq.inverse(circ))

    quasi_inversion_circuit.append(
        gate for gate in reversed(quasi_inverse_gates)
    )

    rand_paulis = cirq.Circuit()
    rand_paulis.append(random_paulis(nqubits))

    circuit = single_qubit_cliffords + forward_circuit
    +rand_paulis + quasi_inversion_circuit
    +cirq.inverse(single_qubit_cliffords)
    for x in range(nqubits):
        circuit.append(cirq.measure(cirq.LineQubit(x)))
    return circuit
