"""Functions to convert from Mitiq's internal circuit representation
to Qiskit representations.
"""

import cirq
from cirq.contrib.qasm_import import circuit_from_qasm
from qiskit import QuantumCircuit


QASM = str


def _to_qasm(circuit: cirq.Circuit) -> QASM:
    """Returns a QASM string representing the input Mitiq circuit.

    Parameters
    ----------
        circuit: Mitiq circuit to convert to a QASM string.

    Returns
    -------
        QASM string equivalent to the input Mitiq circuit.
    """
    return circuit.to_qasm()


def _to_qiskit(circuit: cirq.Circuit) -> QuantumCircuit:
    """Returns a Qiskit circuit equivalent to the input Mitiq circuit.

    Parameters
    ----------
        circuit: Mitiq circuit to convert to a Qiskit circuit.

    Returns
    -------
        Qiskit.QuantumCircuit object equivalent to the input Mitiq circuit.
    """
    return QuantumCircuit.from_qasm_str(_to_qasm(circuit))


def _from_qiskit(circuit: QuantumCircuit) -> cirq.Circuit:
    """Returns a Qiskit circuit equivalent to the input Mitiq circuit.

    Parameters
    ----------
        circuit: Qiskit circuit to convert to a Mitiq circuit.

    Returns
    -------
        Mitiq circuit representation equivalent to the input Qiskit circuit.
    """
    return _from_qasm(circuit.qasm())


def _from_qasm(qasm: QASM) -> cirq.Circuit:
    """Returns a Qiskit circuit equivalent to the input Mitiq circuit.

    Parameters
    ----------
        qasm: QASM string to convert to a Mitiq circuit.

    Returns
    -------
        Mitiq circuit representation equivalent to the input QASM string.
    """
    return circuit_from_qasm(qasm)
