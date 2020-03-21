"""Unit tests for circuit conversions between Mitiq circuits and Qiskit circuits."""

import cirq
import qiskit.extensions.standard as qiskit_ops

from mitiq.qiskit.conversions import (_to_qasm, _to_qiskit, _from_qasm, _from_qiskit)


def test_to_from_bell_state():
    """Minimal test for converting a Bell state circuit."""
    qreg = cirq.LineQubit.range(2)
    cirq_circuit = cirq.Circuit(
        [cirq.ops.H.on(qreg[0]), cirq.ops.CNOT.on(qreg[0], qreg[1])]
    )
    qiskit_circuit = _to_qiskit(cirq_circuit)

    circuit_cirq = _from_qiskit(qiskit_circuit)
    print(circuit_cirq)
    ops = qiskit_circuit.data
    assert len(ops) == 2
    assert isinstance(ops[0], qiskit_ops.HGate)
    assert isinstance(ops[1], qiskit_ops.CnotGate)
    assert False
