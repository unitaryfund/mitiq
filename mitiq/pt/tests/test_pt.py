# Copyright (C) Unitary Fund
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

import random

import cirq
import qiskit
import numpy as np
import networkx as nx

from mitiq.pt.pt import (
    twirl_CNOT_gates,
    twirl_CZ_gates,
    execute_with_pt,
    CNOT_twirling_gates,
    CZ_twirling_gates,
)
from mitiq.interface.mitiq_cirq import compute_density_matrix
from mitiq.benchmarks import generate_mirror_circuit

num_qubits = 2
qubits = cirq.LineQubit.range(num_qubits)
circuit = cirq.Circuit()
circuit.append(cirq.CNOT(*qubits))
circuit.append(cirq.CZ(*qubits))


def amp_damp_executor(circuit: cirq.Circuit, noise: float = 0.005) -> float:
    return compute_density_matrix(
        circuit, noise_model=cirq.amplitude_damp, noise_level=(noise,)
    )[0, 0].real


def test_twirl_CNOT_implements_same_unitary():
    num_circuits = 1
    twirled = twirl_CNOT_gates(circuit, num_circuits=num_circuits)
    assert len(twirled) == num_circuits
    original_unitary = cirq.unitary(circuit)
    twirled_unitary = cirq.unitary(twirled[0])
    assert np.array_equal(twirled_unitary, original_unitary) or np.array_equal(
        -1 * twirled_unitary, original_unitary
    )


def test_twirl_CZ_implements_same_unitary():
    num_circuits = 1
    twirled = twirl_CZ_gates(circuit, num_circuits=num_circuits)
    assert len(twirled) == num_circuits
    original_unitary = cirq.unitary(circuit)
    twirled_unitary = cirq.unitary(twirled[0])
    assert np.array_equal(twirled_unitary, original_unitary) or np.array_equal(
        -1 * twirled_unitary, original_unitary
    )


def test_CNOT_twirl_table():
    a, b = cirq.LineQubit.range(2)
    for P, Q, R, S in CNOT_twirling_gates:
        circuit = cirq.Circuit(
            P.on(a),
            Q.on(b),
            cirq.CNOT.on(a, b),
            R.on(a),
            S.on(b),
            cirq.CNOT.on(a, b),
        )
        assert np.allclose(cirq.unitary(circuit), np.eye(4)) or np.allclose(
            -1 * cirq.unitary(circuit), np.eye(4)
        )


def test_CZ_twirl_table():
    a, b = cirq.LineQubit.range(2)
    for P, Q, R, S in CZ_twirling_gates:
        circuit = cirq.Circuit(
            P.on(a),
            Q.on(b),
            cirq.CZ.on(a, b),
            R.on(a),
            S.on(b),
            cirq.CZ.on(a, b),
        )
        assert np.allclose(cirq.unitary(circuit), np.eye(4)) or np.allclose(
            -1 * cirq.unitary(circuit), np.eye(4)
        )


def test_twirl_CNOT_qiskit():
    qc = qiskit.QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure([0, 1], [0, 1])
    num_circuits = 10
    twirled = twirl_CNOT_gates(qc, num_circuits=num_circuits)
    assert len(twirled) == num_circuits
    random_index = random.randint(0, 9)
    assert isinstance(twirled[random_index], qiskit.QuantumCircuit)


def test_twirl_CNOT_increases_layer_count():
    num_qubits = 3
    num_layers = 10
    gateset = {cirq.X: 1, cirq.Y: 1, cirq.Z: 1, cirq.H: 1, cirq.CNOT: 2}
    circuit = cirq.testing.random_circuit(
        num_qubits, num_layers, op_density=0.8, gate_domain=gateset
    )
    num_CNOTS = sum([op.gate == cirq.CNOT for op in circuit.all_operations()])
    twirled = twirl_CNOT_gates(circuit, num_circuits=1)[0]
    num_gates_before = len(list(circuit.all_operations()))
    num_gates_after = len(list(twirled.all_operations()))
    if num_CNOTS:
        assert num_gates_after > num_gates_before
    else:
        assert num_gates_after == num_gates_before


def test_execute_with_pt():
    num_qubits = 3
    num_layers = 20
    circuit, _ = generate_mirror_circuit(
        nlayers=num_layers,
        two_qubit_gate_prob=1.0,
        connectivity_graph=nx.complete_graph(num_qubits),
    )
    expval = execute_with_pt(
        circuit, amp_damp_executor, num_circuits=10
    )
    assert 0 <= expval < 0.4
