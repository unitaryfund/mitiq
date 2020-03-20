"""Functions to convert from Mitiq's internal circuit representation to supported circuit representations."""

import cirq
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
    pass


def _from_qasm(qasm: QASM) -> cirq.Circuit:
    pass
