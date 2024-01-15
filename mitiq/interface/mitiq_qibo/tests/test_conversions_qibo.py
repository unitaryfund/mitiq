# Copyright (C) Unitary Fund
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for Qibo <-> Cirq conversions."""

import cirq
import numpy as np
import pytest
import qibo
import re 

from mitiq.interface.mitiq_qibo import (
    UnsupportedCirqCircuitError,
    UnsupportedQiboCircuitError,
    from_qibo,
    to_qibo,
    _parse_qasm_modified
)
from mitiq.utils import _equal
from qibo.models.circuit import Circuit as QiboCircuit

def test_from_qibo():
    qibo_circuit = QiboCircuit(2)
    qibo_circuit.add(qibo.gates.CNOT(0, 1))
    qibo_circuit.add(qibo.gates.M(0))

    circuit = from_qibo(qibo_circuit)

    correct = cirq.Circuit(cirq.CNOT(*cirq.LineQubit.range(2)))
    correct.append(cirq.measure(cirq.LineQubit(0)))

    assert _equal(circuit, correct, require_qubit_equality=False)


def test_from_qibo_register_name_error():
    qibo_circuit = QiboCircuit(2)
    qibo_circuit.add(qibo.gates.CNOT(0, 1))
    qibo_circuit.add(qibo.gates.M(0, register_name="K"))

    with pytest.raises(
        UnsupportedQiboCircuitError,
        match="OpenQASM does not support capital letters in"
        + f" register names but K was used.",
    ):
        from_qibo(qibo_circuit)


def test_from_qibo_unsupported_multi_controlled_gate():
    qibo_circuit = QiboCircuit(4)
    qibo_circuit.add(qibo.gates.X(0).controlled_by(1, 2, 3))
    with pytest.raises(
        UnsupportedQiboCircuitError,
        match="OpenQASM does not support multi-controlled gates.",
    ):
        from_qibo(qibo_circuit)


def test_from_qibo_unsupported_gate():
    qibo_circuit = QiboCircuit(2)
    qibo_circuit.add(qibo.gates.fSim(0, 1, 0.4, 0.4))
    with pytest.raises(
        UnsupportedQiboCircuitError,
        match="fsim is not supported by OpenQASM.",
    ):
        from_qibo(qibo_circuit)


def test_from_qibo_unknown_cirq_gate():
    qibo_circuit = QiboCircuit(2)
    qibo_circuit.add(qibo.gates.CRY(0, 1, 0.4))
    qibo_circuit.add(qibo.gates.M(1))
    circuit = from_qibo(qibo_circuit)

    q0, q1 = cirq.LineQubit.range(2)
    correct = cirq.Circuit()
    correct.append(cirq.Ry(rads=0.2).on(q1))
    correct.append(cirq.CNOT(q0, q1))
    correct.append(cirq.Ry(rads=-0.2).on(q1))
    correct.append(cirq.CNOT(q0, q1))
    correct.append(cirq.measure(q1))

    assert _equal(circuit, correct, require_qubit_equality=False)


def test_to_qibo_unsupported_cirq():
    q = cirq.LineQubit(0)
    # Empty circuit
    circuit = cirq.Circuit()

    with pytest.raises(
        UnsupportedCirqCircuitError,
        match="The number of qubits must be at least 1 but is 0.",
    ):
        to_qibo(circuit)


@pytest.mark.parametrize("random_state", range(10))
def test_to_from_qibo(random_state):
    circuit = cirq.testing.random_circuit(
        qubits=4, n_moments=2, op_density=1, random_state=random_state
    )

    converted = from_qibo(to_qibo(circuit))
    # Gates (e.g. iSWAP) aren't guaranteed to be preserved. Check unitary
    # instead of circuit equality.
    cirq.testing.assert_allclose_up_to_global_phase(
        cirq.unitary(converted), cirq.unitary(circuit), atol=1e-7
    )


def test_to_from_qibo_cnot_same_gates():
    qreg = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(cirq.CNOT(*qreg))
    converted = from_qibo(to_qibo(circuit))
    assert _equal(circuit, converted, require_qubit_equality=False)


def test_to_from_qibo_identity():
    q = cirq.LineQubit(0)
    circuit = cirq.Circuit(cirq.I(q))
    # Identity gate
    converted = from_qibo(to_qibo(circuit))
    assert _equal(circuit, converted, require_qubit_equality=False)


@pytest.mark.parametrize("i", range(10))
def test_qibo_integration(i):
    gates = [
        qibo.gates.X(0),
        qibo.gates.Y(0),
        qibo.gates.Z(0),
        qibo.gates.S(0),
        qibo.gates.SDG(0),
        qibo.gates.T(0),
        qibo.gates.I(0),
        qibo.gates.TDG(0),
        qibo.gates.RX(0, 0.4),
        qibo.gates.RY(0, 0.4),
        qibo.gates.RZ(0, 0.4),
        qibo.gates.H(0),
        qibo.gates.TOFFOLI(0, 1, 2),
        qibo.gates.SWAP(0, 1),
        qibo.gates.iSWAP(0, 1),
        qibo.gates.FSWAP(0, 1),
        qibo.gates.CNOT(0, 1),
        qibo.gates.CZ(0, 1),
        qibo.gates.U1(0, 0.4),
        qibo.gates.U2(0, 0.4, 0.5),
        qibo.gates.U3(0, 0.4, 0.5, 0.6),
        qibo.gates.CU1(0, 1, 0.4),
        qibo.gates.CU3(0, 1, 0.4, 0.5, 0.6),
        qibo.gates.CRX(0, 1, 0.4),
        qibo.gates.CRY(0, 1, 0.4),
        qibo.gates.CRZ(0, 1, 0.4),
        qibo.gates.RXX(0, 1, 0.4),
        qibo.gates.RYY(0, 1, 0.4),
        qibo.gates.RZZ(0, 1, 0.4),
    ]

    layers = 3
    np.random.seed(13 * i)
    gates_per_layers = [np.random.permutation(gates) for _ in range(layers)]

    qibo_circuit = QiboCircuit(3)
    for gates in gates_per_layers:
        for gate in gates:
            qibo_circuit.add(gate)

    base_circ = from_qibo(qibo_circuit)
    qibo_recovered = to_qibo(base_circ)
    circ_recovered = from_qibo(qibo_recovered)
    u_1 = cirq.unitary(base_circ)
    u_2 = cirq.unitary(circ_recovered)
    cirq.testing.assert_allclose_up_to_global_phase(u_1, u_2, atol=0)


def test_invalid_QASM_start():
    qasm = "OPENQASM 1.0"
    with pytest.raises(
        ValueError, 
        match="QASM code should start with 'OPENQASM 2.0'.",
    ):
        _parse_qasm_modified(qasm)


def test_invalid_QASM_qubit_arg():
    qasm = "OPENQASM 2.0; \n qreg q[5]; \n ry(pi*-0.5) q[r];" 
    with pytest.raises(
        ValueError, 
        match="Invalid QASM qubit arguments:"
    ):
        _parse_qasm_modified(qasm)


def test_invalid_QASM_measuremnt():
    qasm = "OPENQASM 2.0; \n qreg q[5]; \n measure q[4] -> c[0] -> c[1];" 
    with pytest.raises(
        ValueError, 
        match="Invalid QASM measurement:"
    ):
        _parse_qasm_modified(qasm)


def test_invalid_QASM_qubit_measurment(): 
    qasm = "OPENQASM 2.0; \n qreg q[5]; \n measure q[6] -> c[0];" 
    with pytest.raises(
        ValueError, 
        match=re.escape("Qubit ('q', 6) is not defined in QASM code.")
    ):
        _parse_qasm_modified(qasm)


def test_invalid_QASM_register_measurment(): 
    qasm = "OPENQASM 2.0; \n qreg q[5]; \n measure q[1] -> c[10];" 
    with pytest.raises(
        ValueError, 
        match=re.escape("Classical register name c is not defined in QASM code.")
    ):
        _parse_qasm_modified(qasm)


def test_invalid_QASM_register_measurment(): 
    qasm = "OPENQASM 2.0; \n qreg q[5]; \n measure q[1] -> c[10];" 
    with pytest.raises(
        ValueError, 
        match=re.escape("Classical register name c is not defined in QASM code.")
    ):
        _parse_qasm_modified(qasm)

def test_invalid_QASM_register_index_measurment(): 
    qasm = "OPENQASM 2.0; \n qreg q[5]; creg c[3]; \n measure q[1] -> c[10];" 
    with pytest.raises(
        ValueError, 
        match=re.escape("Cannot access index 10 of register c with 3 qubits.")
    ):
        _parse_qasm_modified(qasm)


def test_invalid_QASM_register_already_used_measurment(): 
    qasm = "OPENQASM 2.0; \n qreg q[5]; creg c[3]; \n measure q[1] -> c[1]; \n measure q[3] -> c[1]" 
    with pytest.raises(
        KeyError, 
        match=re.escape("Key 1 of register c has already been used.")
    ):
        _parse_qasm_modified(qasm)

def test_invalid_QASM_gate(): 
    qasm = "OPENQASM 2.0; \n qreg q[5]; \n hi q[1]" 
    with pytest.raises(
        ValueError, 
        match=re.escape("QASM command hi is not recognized.")
    ):
        _parse_qasm_modified(qasm)

def test_invalid_QASM_missing_parameters_gate(): 
    qasm = "OPENQASM 2.0; \n qreg q[5]; \n rx q[1]" 
    with pytest.raises(
        ValueError, 
        match=re.escape("Missing parameters for QASM gate rx.")
    ):
        _parse_qasm_modified(qasm)

def test_invalid_QASM_parametrized_gate(): 
    qasm = "OPENQASM 2.0; \n qreg q[5]; \n parametrised_gate(0.4) q[1]," 
    with pytest.raises(
        ValueError, 
        match=re.escape("Invalid QASM command parametrised_gate(0.4).")
    ):
        _parse_qasm_modified(qasm)

def test_invalid_QASM_value_parametrized_gate(): 
    qasm = "OPENQASM 2.0; \n qreg q[5]; \n rx(0.4j) q[1]," 
    with pytest.raises(
        ValueError, 
        match=re.escape("Invalid value ['0.4j'] for gate parameters.")
    ):
        _parse_qasm_modified(qasm)

def test_invalid_QASM_2_parameters_gate(): 
    qasm = "OPENQASM 2.0; \n qreg q[5]; \n multi(0.4)(0.5) q[1]," 
    with pytest.raises(
        ValueError, 
        match=re.escape("QASM command multi(0.4)(0.5) is not recognized.")
    ):
        _parse_qasm_modified(qasm)


def test_invalid_QASM_qubit_gate(): 
    qasm = "OPENQASM 2.0; \n qreg q[5]; \n h q[7]" 
    with pytest.raises(
        ValueError, 
        match=re.escape("Qubit ('q', 7) is not defined in QASM code.")
    ):
        _parse_qasm_modified(qasm)




