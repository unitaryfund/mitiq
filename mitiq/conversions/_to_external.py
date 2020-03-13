"""Functions to convert from Mitiq's internal circuit representation to supported circuit representations."""

from typing import Tuple

import cirq
from qiskit import (QuantumCircuit, QuantumRegister, ClassicalRegister)


def _to_qiskit(circuit: cirq.Circuit) -> Tuple[QuantumRegister, ClassicalRegister, QuantumCircuit]:
    """Converts internal representation to Qiskit circuits."""
    pass
