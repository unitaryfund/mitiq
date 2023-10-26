# Copyright (C) Unitary Fund
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

"""Functions to convert between Mitiq's internal circuit representation and
Qibo's circuit representation.
"""

from cirq import Circuit
from qibo import Circuit as QiboCircuit
from pennylane_qiskit.qiskit_device import QISKIT_OPERATION_MAP

from mitiq.interface.mitiq_qiskit import from_qasm as cirq_from_qasm
from mitiq.interface.mitiq_qiskit import to_qasm as cirq_to_qasm
from pennylane import from_qasm as pennylane_from_qasm
from pennylane.tape import QuantumTape
from pennylane.wires import Wires

UNSUPPORTED_QIBO = {"crx", "cry", "crz"}
SUPPORTED_PL = set(QISKIT_OPERATION_MAP.keys())
UNSUPPORTED = {"CRX", "CRY", "CRZ", "S", "T"}
SUPPORTED = SUPPORTED_PL - UNSUPPORTED


class UnsupportedCirqCircuitError(Exception):
    pass

class UnsupportedQiboCircuitError(Exception): 
    pass 


def from_qibo(qibo_circuit: QiboCircuit) -> Circuit:
    """Returns a Cirq circuit equivalent to the input QiboCircuit.

    Args:
        qibo_circuit: QiboCircuit to convert to a Cirq circuit.

    Returns:
        Cirq circuit representation equivalent to the input QiboCircuit.
    """
    for measurement in qibo_circuit.measurements:
        reg_name = measurement.register_name
        reg_qubits = measurement.target_qubits
        if not reg_name.islower():
            raise UnsupportedQiboCircuitError(
                f"OpenQASM does not support capital letters in register names but {reg_name} was used."
                )

    for gate in qibo_circuit.queue:
        if gate.is_controlled_by:
            raise UnsupportedQiboCircuitError(
                "OpenQASM does not support multi-controlled gates."
                )

    
    qasm = qibo_circuit.to_qasm()

    
    unsupported_gate = False
    for gate in qibo_circuit.queue: 
        if gate.name in UNSUPPORTED_QIBO:
            unsupported_gate = True 
            
    #Convert through PennyLane if the circuit contains an unsupported gate
    if unsupported_gate: 

        #Pennylane does not support measurments in the input tape 
        for gate in qibo_circuit.queue:
            if gate.name == 'measure': 
                    raise UnsupportedQiboCircuitError(
                        "One or more unsupported gates CRX, CRY or CRZ were included in a circuit with measurements. "
                        "In this case, measurments should be subsequently added by the executor."
                        )

        qfunc = pennylane_from_qasm(qasm)
        with QuantumTape() as tape:
            qfunc(wires=Wires(range(qibo_circuit.nqubits)))
        tape = tape.expand(stop_at=lambda obj: obj.name in SUPPORTED)
        qasm = tape.to_openqasm(rotations=False, wires=sorted(tape.wires), measure_all=False)

    return cirq_from_qasm(qasm)


def to_qibo(circuit: Circuit) -> QiboCircuit:
    """Returns a QiboCircuit equivalent to the input Cirq circuit.

    Args:
        circuit: Cirq circuit to convert to a QiboCircuit.

    Returns:
        QiboCircuit object equivalent to the input Mitiq circuit.
    """
    qasm = cirq_to_qasm(circuit)
    nqubits, gate_list = QiboCircuit._parse_qasm(qasm)
    
    if not isinstance(nqubits, int):
        raise UnsupportedCirqCircuitError(
        f"The number of qubits must be an integer but is {nqubits}."
        )
    if nqubits < 1:
        raise UnsupportedCirqCircuitError(
        f"The number of qubits must be at least 1 but is {nqubits}."
        )
               
    return QiboCircuit.from_qasm(qasm)

