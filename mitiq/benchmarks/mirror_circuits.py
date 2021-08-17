#!/usr/bin/env python
# coding: utf-8
# %%

# %%


import cirq
import numpy as np
from numpy import random
import networkx as nx
from typing import Optional
from cirq.experiments.qubit_characterizations import _single_qubit_cliffords


# %%
paulis = [cirq.X, cirq.Y, cirq.Z, cirq.I]

def random_paulis(nqubits: int) -> cirq.Circuit:
    """Returns a circuit with a random pauli gate applied to each qubit

    Args:
    nqubits: The number of qubits in the circuit
    """
    circuit = cirq.Circuit()
    circuit.append((paulis[random.randint(4)](cirq.LineQubit(x)) for x in range(nqubits)))
    return circuit


# %%


def edge_grab(xi: float, connectivity_graph_: nx.Graph) -> nx.Graph:
    """Returns a set of edges for which two qubit gates are to be applied given a two qubit gate density, xi,
    and the connectivity graph that must be satisfied

    Args:
        xi: Two qubit gate density
        connectivity_graph_: The connectivity graph for the backend on which the circuit will be run
    """
    connectivity_graph = nx.Graph()
    connectivity_graph.add_nodes_from(connectivity_graph_)
    connectivity_graph.add_edges_from(connectivity_graph_.edges)
    candidate_edges = nx.Graph()
    
    nqubits = connectivity_graph.number_of_nodes()

    while(len(list(connectivity_graph.edges))>0):
        num = random.randint(connectivity_graph.size())
        edges = list(connectivity_graph.edges)
        curr_edge = edges[num]
        q1, q2 = curr_edge
        candidate_edges.add_edge(*curr_edge)
        connectivity_graph.remove_node(q1)
        connectivity_graph.remove_node(q2)
    prob = nqubits*xi / len(list(candidate_edges.edges))

    final_edges = nx.Graph()
    for edge in list(candidate_edges.edges):
        if(random.random()<prob):
            final_edges.add_edge(*edge)
    return final_edges


# %%


single_q_cliffords = _single_qubit_cliffords()


# %%


def random_cliffords(nqubits: int, edges: list) -> cirq.Circuit:
    """Returns a circuit with a two qubit clifford gate applied to each edge in edges, and a random single qubit
    clifford gate applied to every other qubit

    Args:
        nqubits: The number of qubits in the circuit
        edges: The edges for which the two qubit clifford gate is to be applied
    """
    cliffords = single_q_cliffords.c1_in_xy

    gates = [cirq.CNOT(cirq.LineQubit(a), cirq.LineQubit(b)) for a,b in edges]

    for x in range (nqubits):
        notEdge = True
        for a,b in edges:
            if(a==x or b==x):
                notEdge = False
        if(notEdge):
            num = random.randint(24)
            for clifford_gate in cliffords[num]:
                gates.append(clifford_gate(cirq.LineQubit(x)))
    return cirq.Circuit(gates)


# %%


def random_single_cliffords(nqubits: int) -> cirq.Circuit:
    """Returns a layer with a random single qubit clifford gate applied on each qubit
    
    Args:
        nqubits: The number of qubits in the circuit

    """
        
    circuit = cirq.Circuit()
    cliffords =  single_q_cliffords.c1_in_xy
    gates = []
    for x in range(nqubits):
        num = random.randint(24)
        for clifford_gate in cliffords[num]:
            gates.append(clifford_gate(cirq.LineQubit(x)))
    circuit.append((gate for gate in gates))
    return circuit


# %%


def mirror_circuit(depth: int, xi: float, connectivity_graph: nx.Graph) -> cirq.Circuit:
    """Returns a randomized mirror circuit
    
    Args:
        depth: The number of random clifford and pauli layers in the mirror circuit including the inverse layers,
          but excluding the central pauli layer and the initial clifford layer and its inverse
        xi: Expected two-qubit gate density of the mirror circuit
        connectivity_graph: The connectivity graph of the backend on which the mirror circuit will be run. This is
            used to make sure 2 qubit gates are only run on connected qubits
      """
    if not 0 <= xi <= 1:
        raise ValueError("xi should be between 0 and 1")
    
    if not depth%4 == 0:
        raise ValueError("depth must be a multiple of 4")
        
    nqubits = connectivity_graph.number_of_nodes()

    single_qubit_cliffords = cirq.Circuit()
    single_qubit_cliffords.append(random_single_cliffords(nqubits))
    
    forward_circuit = cirq.Circuit()
    
    quasi_inversion_circuit = cirq.Circuit()
    quasi_inverse_gates = []
    
    for _ in range(int(depth/4)):
        forward_circuit.append(random_paulis(nqubits))

        selected_edges = edge_grab(xi, connectivity_graph)
            
        circ = random_cliffords(nqubits, list(selected_edges.edges))
        forward_circuit.append(circ)
        
        quasi_inverse_gates.append(random_paulis(nqubits))
        quasi_inverse_gates.append(cirq.inverse(circ))
    
    
    quasi_inversion_circuit.append(gate for gate in reversed(quasi_inverse_gates))
        
    rand_paulis = cirq.Circuit()
    rand_paulis.append(random_paulis(nqubits))
    
    circuit = single_qubit_cliffords.moments + forward_circuit.moments + rand_paulis + quasi_inversion_circuit + cirq.inverse(single_qubit_cliffords).moments
    for x in range (nqubits):
        circuit.append(cirq.measure(cirq.LineQubit(x)))
    return circuit

