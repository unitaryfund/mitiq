# Copyright (C) Unitary Fund
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

"""Functions to convert between Mitiq's internal circuit representation and
Qibo's circuit representation.
"""
import re
from cirq import Circuit, decompose
from numpy import pi
from qibo import gates
from qibo.gates.abstract import Gate
from qibo.models.circuit import Circuit as QiboCircuit
from qibo.config import raise_error
from typing import cast,Tuple, List, Generator, Union, Optional, Dict

from mitiq.interface.mitiq_qiskit import from_qasm as cirq_from_qasm
from mitiq.interface.mitiq_qiskit import to_qasm as cirq_to_qasm


def crx_decomp(gate: gates.CRX) -> List[Gate]:
    """Decomposes CRX gate to Cirq known gates.

    Args:
        qibo_gate: CRX gate to decompose.

    Returns:
        List with gates that has the same effect as applying the original gate.
    """
    q0, q1 = gate.init_args
    theta = gate.parameters[0]
    decomp_gate = [
        gates.RZ(q1, pi / 2),
        gates.RY(q1, theta / 2),
        gates.CNOT(q0, q1),
        gates.RY(q1, -theta / 2),
        gates.CNOT(q0, q1),
        gates.RZ(q1, -pi / 2),
    ]
    return decomp_gate


def cry_decomp(gate: gates.CRY) -> List[Gate]:
    """Decomposes CRY gate to Cirq known gates.

    Args:
        qibo_gate: CRY gate to decompose.

    Returns:
        List with gates that has the same effect as applying the original gate.
    """
    q0, q1 = gate.init_args
    theta = gate.parameters[0]
    decomp_gate = [
        gates.RY(q1, theta / 2),
        gates.CNOT(q0, q1),
        gates.RY(q1, -theta / 2),
        gates.CNOT(q0, q1),
    ]
    return decomp_gate


def crz_decomp(gate: gates.CRZ) -> List[Gate]:
    """Decomposes CRZ gate to Cirq known gates.

    Args:
        gate: CRZ gate to decompose.

    Returns:
        List with gates that has the same effect as applying the original gate.
    """
    q0, q1 = gate.init_args
    theta = gate.parameters[0]
    decomp_gate = [
        gates.RZ(q1, theta / 2),
        gates.CNOT(q0, q1),
        gates.RZ(q1, -theta / 2),
        gates.CNOT(q0, q1),
    ]
    return decomp_gate


def cu1_decomp(gate: gates.CU1) -> List[Gate]:
    """Decomposes CU1 gate to Cirq known gates.

    Args:
        gate: CU1 gate to decompose.

    Returns:
        List with gates that has the same effect as applying the original gate.
    """
    q0, q1 = gate.init_args
    theta = gate.parameters[0]
    decomp_gate = [
        gates.U1(q0, theta / 2),
        gates.CNOT(q0, q1),
        gates.U1(q1, -theta / 2),
        gates.CNOT(q0, q1),
        gates.U1(q1, theta / 2),
    ]
    return decomp_gate


def cu3_decomp(gate: gates.CU3) -> List[Gate]:
    """Decomposes CU3 gate to Cirq known gates.

    Args:
        gate: CU3 gate to decompose.

    Returns:
        List with gates that has the same effect as applying the original gate.
    """
    q0, q1 = gate.init_args
    theta = gate.parameters[0]
    phi = gate.parameters[1]
    lam = gate.parameters[2]
    decomp_gate = [
        gates.U1(q0, lam / 2 + phi / 2),
        gates.U1(q1, lam / 2 - phi / 2),
        gates.CNOT(q0, q1),
        gates.U3(q1, -theta / 2, 0, -lam / 2 - phi / 2),
        gates.CNOT(q0, q1),
        gates.U3(q1, theta / 2, phi, 0),
    ]
    return decomp_gate


def iswap_decomp(gate: gates.iSWAP) -> List[Gate]:
    """Decomposes ISWAP gate to Cirq known gates.

    Args:
        gate: ISWAP gate to decompose.

    Returns:
        List with gates that has the same effect as applying the original gate.
    """
    q0, q1 = gate.init_args
    decomp_gate = [
        gates.S(q1),
        gates.S(q0),
        gates.H(q0),
        gates.CNOT(q0, q1),
        gates.CNOT(q1, q0),
        gates.H(q1),
    ]
    return decomp_gate


def fswap_decomp(gate: gates.FSWAP) -> List[Gate]:
    """Decomposes FSWAP gate to Cirq known gates.

    Args:
        gate: FSWAP gate to decompose.

    Returns:
        List with gates that has the same effect as applying the original gate.
    """
    q0, q1 = gate.init_args
    decomp_gate = [
        gates.H(q1),
        gates.H(q0),
        gates.CNOT(q1, q0),
        gates.RZ(q0, pi / 2),
        gates.CNOT(q1, q0),
        gates.H(q1),
        gates.H(q0),
        gates.RX(q0, pi / 2),
        gates.RX(q1, pi / 2),
        gates.CNOT(q1, q0),
        gates.RZ(q0, pi / 2),
        gates.CNOT(q1, q0),
        gates.RX(q0, -pi / 2),
        gates.RX(q1, -pi / 2),
        gates.RZ(q0, pi / 2),
        gates.RZ(q1, pi / 2),
    ]
    return decomp_gate


def rxx_decomp(gate: gates.RXX) -> List[Gate]:
    """Decomposes RXX gate to Cirq known gates.

    Args:
        gate: RXX gate to decompose.

    Returns:
        List with gates that has the same effect as applying the original gate.
    """
    q0, q1 = gate.init_args
    theta = gate.parameters[0]
    decomp_gate = [
        gates.H(q0),
        gates.H(q1),
        gates.CNOT(q0, q1),
        gates.RZ(q1, theta),
        gates.CNOT(q0, q1),
        gates.H(q0),
        gates.H(q1),
    ]
    return decomp_gate


def ryy_decomp(gate: gates.RYY) -> List[Gate]:
    """Decomposes RYY gate to Cirq known gates.

    Args:
        gate: RYY gate to decompose.

    Returns:
        List with gates that has the same effect as applying the original gate.
    """
    q0, q1 = gate.init_args
    theta = gate.parameters[0]
    decomp_gate = [
        gates.RX(q0, pi / 2),
        gates.RX(q1, pi / 2),
        gates.CNOT(q0, q1),
        gates.RZ(q1, theta),
        gates.CNOT(q0, q1),
        gates.RX(q0, -pi / 2),
        gates.RX(q1, -pi / 2),
    ]
    return decomp_gate


def rzz_decomp(gate: gates.RZZ) -> List[Gate]:
    """Decomposes RZZ gate to Cirq known gates.

    Args:
        gate: RZZ gate to decompose.

    Returns:
        List with gates that has the same effect as applying the original gate.
    """
    q0, q1 = gate.init_args
    theta = gate.parameters[0]
    decomp_gate = [
        gates.CNOT(q0, q1),
        gates.RZ(q1, theta),
        gates.CNOT(q0, q1),
    ]
    return decomp_gate


GATES_TO_DECOMPOSE = {
    "crx": crx_decomp,
    "cry": cry_decomp,
    "crz": crz_decomp,
    "cu1": cu1_decomp,
    "cu3": cu3_decomp,
    "iswap": iswap_decomp,
    "fswap": fswap_decomp,
    "rxx": rxx_decomp,
    "ryy": ryy_decomp,
    "rzz": rzz_decomp,
}


class UnsupportedCirqCircuitError(Exception):
    pass


class UnsupportedQiboCircuitError(Exception):
    pass


def decompose_qibo_circuit(qibo_circuit: QiboCircuit) -> QiboCircuit:
    """Returns a QiboCircuit circuit equivalent to the input QiboCircuit
    with all unknown by cirq gates decomposed.

    Args:
        qibo_circuit: QiboCircuit to decompose.

    Returns:
        Decomposed QiboCircuit
    """
    decomp_circuit = QiboCircuit(qibo_circuit.nqubits)
    for gate in qibo_circuit.queue:
        if gate.name in GATES_TO_DECOMPOSE:
            function = GATES_TO_DECOMPOSE[gate.name]
            decomposed_gate = function(gate)
            decomp_circuit.add(decomposed_gate)
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
        if isinstance(gate, gates.M):
            continue
        if gate.is_controlled_by:
            raise UnsupportedQiboCircuitError(
                "OpenQASM does not support multi-controlled gates."
            )
        if gate.name not in gates.QASM_GATES:
            raise UnsupportedQiboCircuitError(
                f"{gate.name} is not supported by OpenQASM."
            )

    # Decompose the circuit in known Cirq gates
    qasm = decompose_qibo_circuit(qibo_circuit).to_qasm()

    return cirq_from_qasm(qasm)


def to_qibo(circuit: Circuit) -> QiboCircuit:
    """Returns a QiboCircuit equivalent to the input Cirq circuit.

    Args:
        circuit: Cirq circuit to convert to a QiboCircuit.

    Returns:
        QiboCircuit object equivalent to the input Mitiq circuit.
    """
    
    try:  
        qasm = cirq_to_qasm(circuit)
        nqubits, gate_list = _parse_qasm_modified(qasm)
        
    except ValueError: 
        decomp_circuit = Circuit(decompose(circuit)) 
        qasm = cirq_to_qasm(decomp_circuit)
        nqubits, gate_list = _parse_qasm_modified(qasm)
    
    if nqubits < 1:
        raise UnsupportedCirqCircuitError(
            f"The number of qubits must be at least 1 but is {nqubits}."
        )

    qibo_circuit = QiboCircuit(nqubits)
    for gate_name, qubits, params in gate_list:
        gate = getattr(gates, gate_name)
        if gate_name == "M":
            qibo_circuit.add(gate(*qubits, register_name=params))
        elif params is None:
            qibo_circuit.add(gate(*qubits))
        else:
            # assume parametrized gate
            qibo_circuit.add(gate(*qubits, *params))
    
    return qibo_circuit


def read_args(args: str) ->  Generator[Tuple[str, int], None, None]:
        _args = iter(re.split(r"[\[\],]", args))
        for name in _args:
            if name:
                index = next(_args)
                if not index.isdigit():
                    raise_error(ValueError, "Invalid QASM qubit arguments:", args)
                yield name, int(index)


def _parse_qasm_modified(qasm_code: str) -> Tuple[int, List[Tuple[str, List[int], Union[List[float], str, None]]]]:
    
    """Extracts circuit information from QASM script.

    Args:
        qasm_code: String with the QASM code to parse.

    Returns:
        nqubits: The total number of qubits in the circuit.
        gate_list: List that specifies the gates of the circuit.
            Contains tuples of the form
            (Qibo gate name, qubit IDs, optional additional parameter).
            The additional parameter is the ``register_name`` for
            measurement gates or ``theta`` for parametrized gates.
    """
     
    lines = (line for line in "".join(
        line for line in qasm_code.split("\n") if line and line[:2] != "//" and line[1:3] != "//"
    ).split(";") if line and "//" not in line)

    if next(lines) != "OPENQASM 2.0":
        raise_error(ValueError, "QASM code should start with 'OPENQASM 2.0'.")

    qubits: Dict[Tuple[str, int], int] = {}  
    cregs_size = {}  
    registers: Dict[str, Optional[Dict[int, int]]] = {}
    gate_list: List[Tuple[str, List[int], Union[List[float], str, None]]] = []
    
    for line in lines:
        command, args = line.split(None, 1)
        # remove spaces
        command = command.replace(" ", "")
        args = args.replace(" ", "")

        if command == "include":
            pass

        elif command == "qreg":
            for name, nqubits in read_args(args):
                for i in range(nqubits):
                    qubits[(name, i)] = len(qubits)

        elif command == "creg":
            for name, nqubits in read_args(args):
                cregs_size[name] = nqubits

        elif command == "measure":
            args_list = args.split("->")
            if len(args_list) != 2:
                raise_error(ValueError, "Invalid QASM measurement:", line)
            qubit = next(read_args(args_list[0]))
            if qubit not in qubits:
                raise_error(
                    ValueError,
                    "Qubit {} is not defined in QASM code." "".format(qubit),
                )

            register, idx = next(read_args(args_list[1]))
            if register not in cregs_size:
                raise_error(
                    ValueError,
                    "Classical register name {} is not defined "
                    "in QASM code.".format(register),
                )
            if idx >= cregs_size[register]:
                raise_error(
                    ValueError,
                    "Cannot access index {} of register {} "
                    "with {} qubits."
                    "".format(idx, register, cregs_size[register]),
                )
            if register in registers:
                registers_not_none = {key: value for key, value in registers.items() if value is not None}
                if idx in registers_not_none[register]:
                    raise_error(
                        KeyError,
                        "Key {} of register {} has already "
                        "been used.".format(idx, register),
                    )
                
                if registers[register] is not None:
                    cast(Dict[int, int], registers[register])
                    registers[register][idx] = qubits[qubit]
            else:
                registers[register] = {idx: qubits[qubit]}
                gate_list.append(("M", [0], register))
        else:
            pieces = [x for x in re.split("[()]", command) if x]
            if len(pieces) == 1:
                gatename, params = pieces[0], None
                if gatename not in gates.QASM_GATES:
                    if gatename != "sx" and gatename != "sxdg": 
                        raise_error(
                            ValueError,
                            "QASM command {} is not recognized." "".format(command),
                        )
                if gatename in gates.PARAMETRIZED_GATES:
                    raise_error(
                        ValueError,
                        "Missing parameters for QASM " "gate {}.".format(gatename),
                    )

            elif len(pieces) == 2:
                gatename = pieces[0]
                params = pieces[1].replace(" ", "").split(",")
                if gatename not in gates.PARAMETRIZED_GATES:
                    raise_error(
                        ValueError, "Invalid QASM command {}." "".format(command)
                    )
                try:
                    for i, p in enumerate(params):
                        denominator = 1
                        if "/" in p:
                            p, denominator = p.split("/")
                        if "pi" in p:
                            from functools import reduce
                            from operator import mul

                            s = p.replace("pi", str(pi)).split("*")
                            p = reduce(mul, [float(j) for j in s], 1)
                        params[i] = float(p)
                except ValueError:
                    raise_error(
                        ValueError,
                        "Invalid value {} for gate parameters." "".format(params),
                    )

            else:
                raise_error(
                    ValueError,
                    "QASM command {} is not recognized." "".format(command),
                )

            # Add gate to gate list
            qubit_list = []
            for qubit in read_args(args):
                if qubit not in qubits:
                    raise_error(
                        ValueError,
                        "Qubit {} is not defined in QASM " "code.".format(qubit),
                    )
                qubit_list.append(qubits[qubit])
            if gatename == "sx": 
                gate_list.append(("RX", list(qubit_list),  [pi/2]))
            elif gatename == "sxdg": 
                gate_list.append(("RX", list(qubit_list),  [-pi/2]))
            else: 
                float_params = None if params is None else [float(element) for element in params]
                gate_list.append((str(gates.QASM_GATES[gatename]), list(qubit_list), float_params))
    # Create measurement gate qubit lists from registers
    for i, (gatename, _, third_arg) in enumerate(gate_list):
        if gatename == "M":
            register = third_arg
            qubit_dict = cast(Dict[int, int], registers[register])
            qubit_list = [qubit_dict[k] for k in sorted(qubit_dict.keys())]
            gate_list[i] = ("M", qubit_list, register)
    
    return len(qubits), gate_list

