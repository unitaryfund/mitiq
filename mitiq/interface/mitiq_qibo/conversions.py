# Copyright (C) Unitary Fund
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

"""Functions to convert between Mitiq's internal circuit representation and
Qibo's circuit representation.
"""

from cirq import Circuit
from qibo import Circuit as QiboCircuit
from qibo import gates
from numpy import pi 

from mitiq.interface.mitiq_qiskit import from_qasm as cirq_from_qasm
from mitiq.interface.mitiq_qiskit import to_qasm as cirq_to_qasm


UNSUPPORTED_QIBO = {"crx", "cry", "crz"}

class UnsupportedCirqCircuitError(Exception):
    pass

class UnsupportedQiboCircuitError(Exception): 
    pass 

def decompose_qibo_circuit(qibo_circuit: QiboCircuit) -> QiboCircuit:
    """Returns a QiboCircuit circuit equivalent to the input QiboCircuit
    with all unsupported gates decomposed.

    Args:
        qibo_circuit: QiboCircuit to decompose.

    Returns:
        Decomposed QiboCircuit
    """
    decomp_circuit = qibo_circuit.__class__(qibo_circuit.nqubits)
    for gate in qibo_circuit.queue:
        if gate.name == "crx": 
            q0,q1 = gate.init_args
            theta = gate.parameters[0]
            decomp_circuit.add(gates.RZ(q1,pi/2))
            decomp_circuit.add(gates.RY(q1,theta/2))
            decomp_circuit.add(gates.CNOT(q0,q1)) 
            decomp_circuit.add(gates.RY(q1,-theta/2))
            decomp_circuit.add(gates.CNOT(q0,q1))
            decomp_circuit.add(gates.RZ(q1,-pi/2,0))
        elif gate.name == "cry": 
            q0,q1 = gate.init_args
            theta = gate.parameters[0]
            decomp_circuit.add(gates.RY(q1,theta/2))
            decomp_circuit.add(gates.CNOT(q0,q1)) 
            decomp_circuit.add(gates.RY(q1,-theta/2))
            decomp_circuit.add(gates.CNOT(q0,q1))
        elif gate.name == "crz": 
            q0,q1 = gate.init_args
            theta = gate.parameters[0]
            decomp_circuit.add(gates.RZ(q1,theta/2))
            decomp_circuit.add(gates.CNOT(q0,q1)) 
            decomp_circuit.add(gates.RZ(q1,-theta/2))
            decomp_circuit.add(gates.CNOT(q0,q1))
        else: 
            decomp_circuit.add(gate)

    return decomp_circuit


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

    #Decompose the circuit if it contains an unsupported gate
    unsupported_gate = False
    for gate in qibo_circuit.queue: 
        if gate.name in UNSUPPORTED_QIBO:
            unsupported_gate = True 
    if unsupported_gate: 
        qibo_circuit = decompose_qibo_circuit(qibo_circuit)
        
    qasm = qibo_circuit.to_qasm()

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

