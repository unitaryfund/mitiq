"""Unit tests for circuit conversions between Mitiq circuits and Qiskit circuits."""

import cirq
import qiskit.extensions.standard as qiskit_ops

from mitiq.qiskit.conversions import (_to_qasm, _to_qiskit)


def test_to_bell_state():
    """Minimal test for converting a Bell state circuit."""
    qreg = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(
        [cirq.ops.H.on(qreg[0]), cirq.ops.CNOT.on(qreg[0], qreg[1])]
    )
    circuit = _to_qiskit(circuit)
    ops = circuit.data
    print(ops)
    assert len(ops) == 2
    assert isinstance(ops[0][0], qiskit_ops.HGate)
    assert isinstance(ops[1][0], qiskit_ops.CnotGate)
