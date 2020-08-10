"""Unit tests for circuit conversions between
Mitiq circuits and Qiskit circuits.
"""

import cirq

from mitiq.utils import _equal
from mitiq.mitiq_qiskit.conversions import (
    to_qasm,
    to_qiskit,
    from_qasm,
    from_qiskit,
)


def test_bell_state_to_from_circuits():
    """Tests cirq.Circuit --> qiskit.QuantumCircuit --> cirq.Circuit
     with a Bell state circuit.
    """
    qreg = cirq.LineQubit.range(2)
    cirq_circuit = cirq.Circuit(
        [cirq.ops.H.on(qreg[0]), cirq.ops.CNOT.on(qreg[0], qreg[1])]
    )
    qiskit_circuit = to_qiskit(cirq_circuit)  # Qiskit from Cirq
    circuit_cirq = from_qiskit(qiskit_circuit)  # Cirq from Qiskit
    assert _equal(cirq_circuit, circuit_cirq)


def test_bell_state_to_from_qasm():
    """Tests cirq.Circuit --> QASM string --> cirq.Circuit
     with a Bell state circuit.
    """
    qreg = cirq.LineQubit.range(2)
    cirq_circuit = cirq.Circuit(
        [cirq.ops.H.on(qreg[0]), cirq.ops.CNOT.on(qreg[0], qreg[1])]
    )
    qasm = to_qasm(cirq_circuit)  # Qasm from Cirq
    circuit_cirq = from_qasm(qasm)
    assert _equal(cirq_circuit, circuit_cirq)


def test_random_circuit_to_from_circuits():
    """Tests cirq.Circuit --> qiskit.QuantumCircuit --> cirq.Circuit
    with a random two-qubit circuit.
    """
    cirq_circuit = cirq.testing.random_circuit(
        qubits=2, n_moments=10, op_density=0.99, random_state=1
    )
    qiskit_circuit = to_qiskit(cirq_circuit)
    circuit_cirq = from_qiskit(qiskit_circuit)
    assert cirq.equal_up_to_global_phase(
        cirq_circuit.unitary(), circuit_cirq.unitary()
    )


def test_random_circuit_to_from_qasm():
    """Tests cirq.Circuit --> QASM string --> cirq.Circuit
     with a random one-qubit circuit.
    """
    cirq_circuit = cirq.testing.random_circuit(
        qubits=2, n_moments=10, op_density=0.99, random_state=2
    )
    qasm = to_qasm(cirq_circuit)
    circuit_cirq = from_qasm(qasm)
    assert cirq.equal_up_to_global_phase(
        cirq_circuit.unitary(), circuit_cirq.unitary()
    )
