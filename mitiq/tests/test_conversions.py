# Copyright (C) 2020 Unitary Fund
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Tests for circuit conversions."""
import pytest

import numpy as np

import cirq
from pyquil import Program, gates
import qiskit
from braket.circuits import (
    Circuit as BKCircuit,
    gates as braket_gates,
    Instruction,
)
import pennylane as qml

from mitiq._typing import SUPPORTED_PROGRAM_TYPES
from mitiq.interface import (
    convert_to_mitiq,
    convert_from_mitiq,
    accept_any_qprogram_as_input,
    atomic_one_to_many_converter,
    noise_scaling_converter,
    UnsupportedCircuitError,
)
from mitiq.utils import _equal

# Cirq Bell circuit.
cirq_qreg = cirq.LineQubit.range(2)
cirq_circuit = cirq.Circuit(
    cirq.ops.H.on(cirq_qreg[0]), cirq.ops.CNOT.on(*cirq_qreg)
)

# Qiskit Bell circuit.
qiskit_qreg = qiskit.QuantumRegister(2)
qiskit_circuit = qiskit.QuantumCircuit(qiskit_qreg)
qiskit_circuit.h(qiskit_qreg[0])
qiskit_circuit.cnot(*qiskit_qreg)

# pyQuil Bell circuit.
pyquil_circuit = Program(gates.H(0), gates.CNOT(0, 1))

# Braket Bell circuit.
braket_circuit = BKCircuit(
    [
        Instruction(braket_gates.H(), 0),
        Instruction(braket_gates.CNot(), [0, 1]),
    ]
)

circuit_types = {
    "cirq": cirq.Circuit,
    "qiskit": qiskit.QuantumCircuit,
    "pyquil": Program,
    "braket": BKCircuit,
    "pennylane": qml.tape.QuantumTape,
}


@noise_scaling_converter
def scaling_function(circ: cirq.Circuit, *args, **kwargs) -> cirq.Circuit:
    return circ


@accept_any_qprogram_as_input
def get_wavefunction(circ: cirq.Circuit) -> np.ndarray:
    return circ.final_state_vector()


@atomic_one_to_many_converter
def returns_several_circuits(circ: cirq.Circuit, *args, **kwargs):
    return [circ] * 5


@pytest.mark.parametrize(
    "circuit", (qiskit_circuit, pyquil_circuit, braket_circuit)
)
def test_to_mitiq(circuit):
    converted_circuit, input_type = convert_to_mitiq(circuit)
    assert _equal(converted_circuit, cirq_circuit)
    assert input_type in circuit.__module__


@pytest.mark.parametrize("item", ("circuit", 1, None))
def test_to_mitiq_bad_types(item):
    with pytest.raises(
        UnsupportedCircuitError,
        match="Could not determine the package of the input circuit.",
    ):
        convert_to_mitiq(item)


@pytest.mark.parametrize("to_type", SUPPORTED_PROGRAM_TYPES.keys())
def test_from_mitiq(to_type):
    converted_circuit = convert_from_mitiq(cirq_circuit, to_type)
    circuit, input_type = convert_to_mitiq(converted_circuit)
    assert _equal(circuit, cirq_circuit)
    assert input_type == to_type


@pytest.mark.parametrize(
    "circuit_and_expected",
    [
        (cirq.Circuit(cirq.X.on(cirq.LineQubit(0))), np.array([0, 1])),
        (cirq_circuit, np.array([1, 0, 0, 1]) / np.sqrt(2)),
    ],
)
@pytest.mark.parametrize("to_type", SUPPORTED_PROGRAM_TYPES.keys())
def test_accept_any_qprogram_as_input(circuit_and_expected, to_type):
    circuit, expected = circuit_and_expected
    wavefunction = get_wavefunction(convert_from_mitiq(circuit, to_type))
    assert np.allclose(wavefunction, expected)


@pytest.mark.parametrize(
    "circuit_and_type",
    (
        (qiskit_circuit, "qiskit"),
        (pyquil_circuit, "pyquil"),
        (braket_circuit, "braket"),
    ),
)
def test_converter(circuit_and_type):
    circuit, input_type = circuit_and_type

    # Return the input type
    scaled = scaling_function(circuit)
    assert isinstance(scaled, circuit_types[input_type])

    # Return a Cirq Circuit
    cirq_scaled = scaling_function(circuit, return_mitiq=True)
    assert isinstance(cirq_scaled, cirq.Circuit)
    assert _equal(cirq_scaled, cirq_circuit)


@pytest.mark.parametrize("nbits", [1, 10])
@pytest.mark.parametrize("measure", [True, False])
def test_converter_keeps_register_structure_qiskit(nbits, measure):
    qreg = qiskit.QuantumRegister(nbits)
    creg = qiskit.ClassicalRegister(nbits)
    circ = qiskit.QuantumCircuit(qreg, creg)
    circ.h(qreg)

    if measure:
        circ.measure(qreg, creg)

    scaled = scaling_function(circ)

    assert scaled.qregs == circ.qregs
    assert scaled.cregs == circ.cregs
    assert scaled == circ


@pytest.mark.parametrize("to_type", SUPPORTED_PROGRAM_TYPES.keys())
def test_atomic_one_to_many_converter(to_type):
    circuit = convert_from_mitiq(cirq_circuit, to_type)
    circuits = returns_several_circuits(circuit)
    for circuit in circuits:
        assert isinstance(circuit, circuit_types[to_type])

    circuits = returns_several_circuits(circuit, return_mitiq=True)
    for circuit in circuits:
        assert isinstance(circuit, cirq.Circuit)
