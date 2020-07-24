import warnings
from functools import singledispatch
from typing import Callable, Mapping, Union

import numpy as np
from pyquil.parser import parse
from pyquil.quilbase import (Declare,
                             Gate as PyQuilGate,
                             Measurement as PyQuilMeasurement,
                             Pragma,
                             Reset,
                             ResetQubit)

from cirq import Circuit, LineQubit
from cirq.ops import (CCNOT, CNOT, CSWAP, CZ, CZPowGate, Gate, H, I, ISWAP,
                      ISwapPowGate, MeasurementGate, S, SWAP, T, X, Y, Z,
                      ZPowGate, rx, ry, rz)


class UnsupportedQuilGate(Exception):
    pass


def cphase(param: float) -> CZPowGate:
    """
    PyQuil's CPHASE and Cirq's CZPowGate are the same up to a factor of pi.
    """
    return CZPowGate(exponent=param / np.pi)


def phase(param: float) -> ZPowGate:
    """
    PyQuil's PHASE and Cirq's ZPowGate are the same up to a factor of pi.
    """
    return ZPowGate(exponent=param / np.pi)


def xy(param: float) -> ISwapPowGate:
    """
    PyQuil's XY and Cirq's ISwapPowGate are the same up to a factor of pi.
    """
    return ISwapPowGate(exponent=param / np.pi)


# parameterized gates map to functions that produce Gate constructors
SUPPORTED_GATES: Mapping[str, Union[Gate, Callable[..., Gate]]] = {
    "CCNOT": CCNOT,
    "CNOT": CNOT,
    "CSWAP": CSWAP,
    "CPHASE": cphase,
    "CZ": CZ,
    "PHASE": phase,
    "H": H,
    "I": I,
    "ISWAP": ISWAP,
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


@singledispatch
def op_from_inst(inst: object) -> None:
    """
    Generic function for converting pyQuil instructions to Cirq operations.
    """
    raise TypeError(f"Quil instruction {inst} of type {type(inst)} not currently supported in Cirq.")


@op_from_inst.register(PyQuilGate)
def _(inst: PyQuilGate) -> Gate:
    """
    Convert pyQuil gates to Cirq operations.
    """
    quil_gate_name = inst.name
    quil_gate_params = inst.params
    line_qubits = list(LineQubit(q.index) for q in inst.qubits)

    if quil_gate_name in SUPPORTED_GATES:
        cirq_gate_fn = SUPPORTED_GATES[quil_gate_name]
        if quil_gate_params:
            return cirq_gate_fn(*quil_gate_params)(*line_qubits)
        return cirq_gate_fn(*line_qubits)
    raise UnsupportedQuilGate(f"Quil gate {quil_gate_name} not supported in Cirq.")


@op_from_inst.register(PyQuilMeasurement)
def _(inst: PyQuilMeasurement) -> MeasurementGate:
    """
    Convert pyQuil MEASURE operations to Cirq MeasurementGate objects.
    """
    line_qubit = LineQubit(inst.qubit.index)
    quil_memory_reference = inst.classical_reg.out()
    return MeasurementGate(1, key=quil_memory_reference)(line_qubit)


@op_from_inst.register(Declare)
def _(inst: Declare) -> Circuit:
    """
    Pass when encountering a DECLARE.
    """
    return Circuit()


@op_from_inst.register(Pragma)
def _(inst: Pragma) -> Circuit:
    """
    Pass and warn when encountering a PRAGMA.
    """
    warnings.warn(f"Ignoring unsupported operation {inst}")
    return Circuit()


@op_from_inst.register(Reset)
def _(inst: Reset) -> Circuit:
    """
    Pass and warn when encountering a RESET.
    """
    warnings.warn(f"Ignoring unsupported operation {inst}")
    return Circuit()


@op_from_inst.register(ResetQubit)
def _(inst: ResetQubit) -> Circuit:
    """
    Pass and warn when encountering a RESET q.
    """
    warnings.warn(f"Ignoring unsupported operation {inst}")
    return Circuit()


def circuit_from_quil(quil: str) -> Circuit:
    """
    Convert a Quil program to a Cirq Circuit.
    """
    circuit = Circuit()
    instructions = parse(quil)
    for inst in instructions:
        circuit += op_from_inst(inst)
    return circuit
