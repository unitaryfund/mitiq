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

from mitiq.benchmarks import mirror_circuits
import networkx as nx
import cirq
import pytest
from cirq.experiments.qubit_characterizations import _single_qubit_cliffords

paulis = [cirq.X, cirq.Y, cirq.Z, cirq.I]
single_q_cliffords = _single_qubit_cliffords().c1_in_xy
single_cliffords = []
for gates in single_q_cliffords:
    for gate in gates:
        single_cliffords.append(gate)
cliffords = single_cliffords
cliffords.append(cirq.CNOT)
all_gates = paulis + cliffords
all_gates.append(cirq.measure)


@pytest.mark.parametrize("n", (0, 5))
def test_random_paulis(n):
    circuit = mirror_circuits.random_paulis(nqubits=n)
    assert isinstance(circuit, cirq.Circuit)
    assert len(circuit.all_qubits()) == n
    assert len(list(circuit.all_operations())) == n
    assert set(op.gate for op in circuit.all_operations()).issubset(paulis)


def test_edge_grab():
    nqubits = [9, 10, 11]
    graphs = []
    for qubits in nqubits:
        graphs.append(nx.complete_graph(qubits))
    removed_edges = [
        [(1, 2), (4, 7), (3, 5)],
        [(2, 4), (5, 6), (7, 8)],
        [(3, 4), (5, 6), (8, 9)],
    ]

    for x in range(3):
        graph = graphs[x]
        for edge in removed_edges[x]:
            graph.remove_edge(*edge)
        selected_edges = mirror_circuits.edge_grab(0.3, graph)
        assert set(selected_edges.edges()).issubset(graph.edges())
        assert selected_edges.number_of_nodes() == nqubits[x]
        for node in selected_edges.nodes:
            neighbors = len(list(selected_edges.neighbors(node)))
            assert neighbors == 1 or neighbors == 0


def test_random_cliffords():
    nqubits = [9, 10, 11]
    edges = [
        [(1, 2), (4, 7), (3, 5)],
        [(2, 4), (5, 6), (7, 8)],
        [(3, 4), (5, 6), (8, 9)],
    ]
    for x in range(3):
        graph = nx.Graph()
        graph.add_nodes_from(range(nqubits[x]))
        graph.add_edges_from(edges[x])
        circuit = mirror_circuits.random_cliffords(graph)
        assert isinstance(circuit, cirq.Circuit)
        assert len(circuit.all_qubits()) == nqubits[x]
        two_q_gates = set()
        for a, b in edges[x]:
            two_q_gates.add(cirq.CNOT(cirq.LineQubit(a), cirq.LineQubit(b)))
        assert two_q_gates.issubset(circuit.all_operations())
        assert set(op.gate for op in circuit.all_operations()).issubset(
            cliffords
        )


def test_random_single_cliffords():
    qubits = [[0, 1, 2, 3, 4], [1, 3, 5, 7], [1, 2, 4, 6, 7, 8]]
    for x in range(3):
        graph = nx.Graph()
        graph.add_nodes_from(qubits[x])
        circuit = mirror_circuits.random_single_cliffords(graph)
        assert isinstance(circuit, cirq.Circuit)
        assert set(circuit.all_qubits()).issubset
        ([cirq.LineQubit(qubit) for qubit in qubits[x]])
        assert set(op.gate for op in circuit.all_operations()).issubset(
            single_cliffords
        )


def test_mirror_circuit():
    parameters = [
        (16, 0.3, nx.complete_graph(3)),
        (20, 0.4, nx.complete_graph(4)),
        (24, 0.5, nx.complete_graph(5)),
    ]
    for depth, xi, connectivity_graph in parameters:
        n = connectivity_graph.number_of_nodes()
        circ = mirror_circuits.generate_mirror_circuit(
            depth, xi, connectivity_graph
        )
        assert isinstance(circ, cirq.Circuit)
        assert len(circ.all_qubits()) == n
        result = (
            cirq.Simulator()
            .run(circ, repetitions=1_000)
            .multi_measurement_histogram(keys=circ.all_measurement_keys())
        )
        assert len(result.keys()) == 1
        meas_circ = cirq.Circuit()
        for qubit in range(n):
            meas_circ.append(cirq.measure(cirq.LineQubit(qubit)))
        for moment in circ:
            temp_circ = cirq.Circuit()
            temp_circ.append(moment)
            if temp_circ != meas_circ:
                assert set(
                    op.gate for op in temp_circ.all_operations()
                ).issubset(all_gates)
        assert circ.has_measurements()
        assert circ.are_all_measurements_terminal()
