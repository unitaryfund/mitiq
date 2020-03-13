"""Functions for converting supported input circuits to Mitiq's internal circuit representation."""

from numpy import cumsum

import cirq
from qiskit import (QuantumCircuit, QuantumRegister, ClassicalRegister)

_QISKII_TO_CIRQ = {
    "h": cirq.ops.H,
    "s": cirq.ops.S,
    "t": cirq.ops.T,
    "cx": cirq.ops.CNOT,
    "measure": cirq.ops.measure
}  # Could also use signatures in Qiskit for keys
_QISKIT_SUPPORTED_GATES = set(_QISKII_TO_CIRQ.keys())


def _qiskit_to_cirq(circuit: QuantumCircuit) -> cirq.Circuit:
    """Returns a Cirq circuit equivalent to the input Qiskit circuit.

    Args:
        circuit: Qiskit QuantumCircuit object to be converted to a Cirq circuit.
    """

    qreg_sizes = list(cumsum([0] + [qreg.size for qreg in circuit.qregs[1:]]))
    qreg_names = [qreg.name for qreg in circuit.qregs]
    creg_sizes = list(cumsum([0] + [creg.size for creg in circuit.cregs[1:]]))
    creg_names = [creg.name for creg in circuit.cregs]

    qreg_shifts = dict(zip(qreg_names, qreg_sizes))
    creg_shifts = dict(zip(creg_names, creg_sizes))

    n = len(circuit.qubits)
    cirq_qreg = cirq.LineQubit.range(n)  # Could be different types: GridQubit, CustomQubit
    cirq_circuit = cirq.Circuit()  # Could have a Device

    for op in circuit.data:
        gate, qubits, _ = op
        # Convert from list of Qiskit Qubit's to linear indices
        # Note: If GridQubit's or another type are used, a different indexing (than linear) is necessary
        indices = [qreg_shifts[qubit.register.name] + qubit.index for qubit in qubits]
        cirq_qubits = [cirq_qreg[ii] for ii in indices]

        cirq_circuit.append(
            _QISKII_TO_CIRQ[gate.name](*cirq_qubits)
        )

    return cirq_circuit  # Could also return cirq_qreg
