# Copyright (C) Unitary Fund
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

from mitiq.pt.pt import add_paulis
import cirq

num_qubits = 2
qubits = cirq.LineQubit.range(num_qubits)
circuit = cirq.Circuit()
circuit.append(cirq.CNOT.on_each(zip(qubits, qubits[1:])))


def test_add_paulis():
    twirled = add_paulis(circuit)
    twirled_circuit = cirq.Circuit(
        cirq.X.on(qubits[0]),
        cirq.I.on(qubits[1]),
        cirq.CNOT.on(*qubits),
        cirq.X.on_each(*qubits),
    )

    assert twirled == twirled_circuit
