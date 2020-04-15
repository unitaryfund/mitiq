"""Unit tests for circuit conversions between
Mitiq circuits and Qiskit circuits.
"""

import cirq

from mitiq.utils import (_equal, random_circuit)
from mitiq.mitiq_qiskit.conversions import (_to_qasm,
                                      _to_qiskit,
                                      _from_qasm,
                                      _from_qiskit)


def test_bell_state_to_from_circuits():
    """Tests cirq.Circuit --> qiskit.QuantumCircuit --> cirq.Circuit
     with a Bell state circuit.
    """
    qreg = cirq.LineQubit.range(2)
    cirq_circuit = cirq.Circuit(
        [cirq.ops.H.on(qreg[0]), cirq.ops.CNOT.on(qreg[0], qreg[1])]
    )
    qiskit_circuit = _to_qiskit(cirq_circuit)  # Qiskit from Cirq
    circuit_cirq = _from_qiskit(qiskit_circuit)  # Cirq from Qiskit
    assert _equal(cirq_circuit, circuit_cirq)


def test_bell_state_to_from_qasm():
    """Tests cirq.Circuit --> QASM string --> cirq.Circuit
     with a Bell state circuit.
    """
    qreg = cirq.LineQubit.range(2)
    cirq_circuit = cirq.Circuit(
        [cirq.ops.H.on(qreg[0]), cirq.ops.CNOT.on(qreg[0], qreg[1])]
    )
    qasm = _to_qasm(cirq_circuit)  # Qasm from Cirq
    circuit_cirq = _from_qasm(qasm)
    assert _equal(cirq_circuit, circuit_cirq)


def test_random_circuit_to_from_circuits():
    """Tests cirq.Circuit --> qiskit.QuantumCircuit --> cirq.Circuit
    with a random one-qubit circuit.
    """
    cirq_circuit = random_circuit(depth=20)
    qiskit_circuit = _to_qiskit(cirq_circuit)
    circuit_cirq = _from_qiskit(qiskit_circuit)
    assert _equal(cirq_circuit, circuit_cirq)


def test_random_circuit_to_from_qasm():
    """Tests cirq.Circuit --> QASM string --> cirq.Circuit
     with a random one-qubit circuit.
    """
    cirq_circuit = random_circuit(depth=20)
    qasm = _to_qasm(cirq_circuit)
    circuit_cirq = _from_qasm(qasm)
    assert _equal(cirq_circuit, circuit_cirq)
