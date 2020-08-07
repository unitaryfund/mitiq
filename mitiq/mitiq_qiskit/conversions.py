"""Functions to convert between Mitiq's internal circuit representation
and Qiskit's circuit representation.
"""

import cirq
from cirq.contrib.qasm_import import circuit_from_qasm
from qiskit import QuantumCircuit
from mitiq.utils import _simplify_circuit


QASMType = str


def to_qasm(circuit: cirq.Circuit) -> QASMType:
    """Returns a QASM string representing the input Mitiq circuit.

    Args:
        circuit: Mitiq circuit to convert to a QASM string.

    Returns:
        QASMType: QASM string equivalent to the input Mitiq circuit.
    """
    # simplify individual gates, e.g. H**-1 is simplified to H
    _simplify_circuit(circuit)
    return circuit.to_qasm()


def to_qiskit(circuit: cirq.Circuit) -> QuantumCircuit:
    """Returns a Qiskit circuit equivalent to the input Mitiq circuit.

    Args:
        circuit: Mitiq circuit to convert to a Qiskit circuit.

    Returns:
        Qiskit.QuantumCircuit object equivalent to the input Mitiq circuit.
    """
    return QuantumCircuit.from_qasm_str(to_qasm(circuit))


def from_qiskit(circuit: QuantumCircuit) -> cirq.Circuit:
    """Returns a Mitiq circuit equivalent to the input Qiskit circuit.

    Args:
        circuit: Qiskit circuit to convert to a Mitiq circuit.

    Returns:
        Mitiq circuit representation equivalent to the input Qiskit circuit.
    """
    return from_qasm(circuit.qasm())


def from_qasm(qasm: QASMType) -> cirq.Circuit:
    """Returns a Mitiq circuit equivalent to the input QASM string.

    Args:
        qasm: QASM string to convert to a Mitiq circuit.

    Returns:
        Mitiq circuit representation equivalent to the input QASM string.
    """
    return circuit_from_qasm(qasm)
