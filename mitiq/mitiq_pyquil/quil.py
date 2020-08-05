from typing import Callable, Dict, Union

import numpy as np
from pyquil.parser import parse
from pyquil.quilbase import (
    Declare,
    DefGate,
    Gate as PyQuilGate,
    Measurement as PyQuilMeasurement,
    Pragma,
    Reset,
    ResetQubit,
)

from cirq import Circuit, LineQubit
from cirq.ops import (
    CCNOT,
    CNOT,
    CSWAP,
    CZ,
    CZPowGate,
    Gate,
    H,
    I,
    ISWAP,
    ISwapPowGate,
    MatrixGate,
    MeasurementGate,
    S,
    SWAP,
    T,
    X,
    Y,
    Z,
    ZPowGate,
    rx,
    ry,
    rz,
)


class UnsupportedQuilGate(Exception):
    pass


class UnsupportedQuilInstruction(Exception):
    pass


#
# Functions for converting supported parameterized Quil gates.
#


def cphase(param: float) -> CZPowGate:
    """
    PyQuil's CPHASE and Cirq's CZPowGate are the same up to a factor of pi.
    """
    return CZPowGate(exponent=param / np.pi)


def cphase00(phi: float) -> MatrixGate:
    """
    PyQuil's CPHASE00 gate can be defined using Cirq's MatrixGate.
    """
    cphase00_matrix = np.array(
        [[np.exp(1j * phi), 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    )
    return MatrixGate(cphase00_matrix)


def cphase01(phi: float) -> MatrixGate:
    """
    PyQuil's CPHASE01 gate can be defined using Cirq's MatrixGate.
    """
    cphase01_matrix = np.array(
        [[1, 0, 0, 0], [0, np.exp(1j * phi), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    )
    return MatrixGate(cphase01_matrix)


def cphase10(phi: float) -> MatrixGate:
    """
    PyQuil's CPHASE10 gate can be defined using Cirq's MatrixGate.
    """
    cphase10_matrix = np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, np.exp(1j * phi), 0], [0, 0, 0, 1]]
    )
    return MatrixGate(cphase10_matrix)


def phase(param: float) -> ZPowGate:
    """
    PyQuil's PHASE and Cirq's ZPowGate are the same up to a factor of pi.
    """
    return ZPowGate(exponent=param / np.pi)


def pswap(phi: float) -> MatrixGate:
    """
    PyQuil's PSWAP gate can be defined using Cirq's MatrixGate.
    """
    pswap_matrix = np.array(
        [
            [1, 0, 0, 0],
            [0, 0, np.exp(1j * phi), 0],
            [0, np.exp(1j * phi), 0, 0],
            [0, 0, 0, 1],
        ]
    )
    return MatrixGate(pswap_matrix)


def xy(param: float) -> ISwapPowGate:
    """
    PyQuil's XY and Cirq's ISwapPowGate are the same up to a factor of pi.
    """
    return ISwapPowGate(exponent=param / np.pi)


PRAGMA_ERROR = """
Please remove PRAGMAs from your Quil program.
If you would like to add noise, do so after conversion.
"""

RESET_ERROR = """
Please remove RESETs from your Quil program.
RESET directives have special meaning on QCS, to enable active reset.
"""

# Parameterized gates map to functions that produce Gate constructors.
SUPPORTED_GATES: Dict[str, Union[Gate, Callable[..., Gate]]] = {
    "CCNOT": CCNOT,
    "CNOT": CNOT,
    "CSWAP": CSWAP,
    "CPHASE": cphase,
    "CPHASE00": cphase00,
    "CPHASE01": cphase01,
    "CPHASE10": cphase10,
    "CZ": CZ,
    "PHASE": phase,
    "H": H,
    "I": I,
    "ISWAP": ISWAP,
    "PSWAP": pswap,
    "RX": rx,
    "RY": ry,
    "RZ": rz,
    "S": S,
    "SWAP": SWAP,
    "T": T,
    "X": X,
    "Y": Y,
    "Z": Z,
    "XY": xy,
}


def circuit_from_quil(quil: str) -> Circuit:
    """
    Convert a Quil program to a Cirq Circuit.
    """
    circuit = Circuit()
    defgates = {}
    instructions = parse(quil)

    for inst in instructions:
        # Add DEFGATE-defined gates to defgates dict using MatrixGate.
        if isinstance(inst, DefGate):
            if inst.parameters:
                raise UnsupportedQuilInstruction(
                    "Parameterized DEFGATEs are currently unsupported."
                )
            defgates[inst.name] = MatrixGate(inst.matrix)

        # Pass when encountering a DECLARE.
        elif isinstance(inst, Declare):
            pass

        # Convert pyQuil gates to Cirq operations.
        elif isinstance(inst, PyQuilGate):
            quil_gate_name = inst.name
            quil_gate_params = inst.params
            line_qubits = list(LineQubit(q.index) for q in inst.qubits)
            defgates_and_supported_gates = dict(**defgates, **SUPPORTED_GATES)
            if quil_gate_name not in defgates_and_supported_gates:
                raise UnsupportedQuilGate(
                    f"Quil gate {quil_gate_name} not supported in Cirq."
                )
            cirq_gate_fn = defgates_and_supported_gates[quil_gate_name]
            if quil_gate_params:
                circuit += cirq_gate_fn(*quil_gate_params)(*line_qubits)
            else:
                circuit += cirq_gate_fn(*line_qubits)

        # Convert pyQuil MEASURE operations to Cirq MeasurementGate objects.
        elif isinstance(inst, PyQuilMeasurement):
            line_qubit = LineQubit(inst.qubit.index)
            quil_memory_reference = inst.classical_reg.out()
            circuit += MeasurementGate(1, key=quil_memory_reference)(line_qubit)

        # Raise a targeted error when encountering a PRAGMA.
        elif isinstance(inst, Pragma):
            raise UnsupportedQuilInstruction(PRAGMA_ERROR)

        # Raise a targeted error when encountering a RESET.
        elif isinstance(inst, (Reset, ResetQubit)):
            raise UnsupportedQuilInstruction(RESET_ERROR)

        # Raise a general error when encountering an unconsidered type.
        else:
            raise UnsupportedQuilInstruction(
                f"Quil instruction {inst} of type {type(inst)}"
                " not currently supported in Cirq."
            )

    return circuit
