# Copyright (C) 2021 Unitary Fund
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

"""Unit tests for Pennylane <-> Cirq conversions."""

import pytest
import numpy as np

import cirq
import pennylane as qml

from mitiq.interface.mitiq_pennylane import (
    from_pennylane,
    to_pennylane,
    UnsupportedQuantumTapeError,
)
from mitiq.utils import _equal


def test_from_pennylane():
    with qml.tape.QuantumTape() as tape:
        qml.CNOT(wires=[0, 1])

    circuit = from_pennylane(tape)
    correct = cirq.Circuit(cirq.CNOT(*cirq.LineQubit.range(2)))

    assert _equal(circuit, correct, require_qubit_equality=False)


def test_from_pennylane_unsupported_tapes():
    with qml.tape.QuantumTape() as tape:
        qml.CZ(wires=[0, "a"])

    with pytest.raises(UnsupportedQuantumTapeError, match="could not sort"):
        from_pennylane(tape)


def test_no_variance():
    with qml.tape.QuantumTape() as tape:
        qml.CNOT(wires=[0, 1])
        qml.expval(qml.PauliZ(0))

    with pytest.raises(
        UnsupportedQuantumTapeError,
        match="Measurements are not supported on the input tape.",
    ):
        from_pennylane(tape)


@pytest.mark.parametrize("random_state", range(10))
def test_to_from_pennylane(random_state):
    circuit = cirq.testing.random_circuit(
        qubits=4, n_moments=2, op_density=1, random_state=random_state
    )

    converted = from_pennylane(to_pennylane(circuit))
    # Gates (e.g. iSWAP) aren't guaranteed to be preserved. Check unitary
    # instead of circuit equality.
    cirq.testing.assert_allclose_up_to_global_phase(
        cirq.unitary(converted), cirq.unitary(circuit), atol=1e-7
    )


def test_to_from_pennylane_cnot_same_gates():
    qreg = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(cirq.CNOT(*qreg))
    converted = from_pennylane(to_pennylane(circuit))
    assert _equal(circuit, converted, require_qubit_equality=False)


def test_to_from_pennylane_identity():
    q = cirq.LineQubit(0)
    # Empty circuit
    circuit = cirq.Circuit()
    converted = from_pennylane(to_pennylane(circuit))
    assert _equal(circuit, converted, require_qubit_equality=False)
    circuit = cirq.Circuit(cirq.I(q))
    # Identity gate
    converted = from_pennylane(to_pennylane(circuit))
    # TODO: test circuit equality after Identity operation will be added
    # to PennyLane (https://github.com/PennyLaneAI/pennylane/issues/1632)
    assert np.allclose(cirq.unitary(circuit), cirq.unitary(converted))


def test_non_consecutive_wires_error():
    with qml.tape.QuantumTape() as tape:
        qml.CNOT(wires=[0, 2])
    with pytest.raises(
        UnsupportedQuantumTapeError, match="contiguously pack",
    ):
        from_pennylane(tape)


def test_integration():
    n_wires = 4

    gates = [
        qml.PauliX,
        qml.PauliY,
        qml.PauliZ,
        qml.S,
        qml.T,
        qml.RX,
        qml.RY,
        qml.RZ,
        qml.Hadamard,
        qml.Rot,
        qml.CRot,
        qml.Toffoli,
        qml.SWAP,
        qml.CSWAP,
        qml.U1,
        qml.U2,
        qml.U3,
        qml.CRX,
        qml.CRY,
        qml.CRZ,
    ]

    layers = 3
    np.random.seed(1967)
    gates_per_layers = [np.random.permutation(gates) for _ in range(layers)]

    with qml.tape.QuantumTape() as tape:
        np.random.seed(1967)
        for gates in gates_per_layers:
            for gate in gates:
                params = list(np.pi * np.random.rand(gate.num_params))
                rnd_wires = np.random.choice(
                    range(n_wires), size=gate.num_wires, replace=False
                )
                gate(
                    *params,
                    wires=[
                        int(w) for w in rnd_wires
                    ],  # make sure we do not address wires as 0-d arrays
                )

    base_circ = from_pennylane(tape)
    tape_recovered = to_pennylane(base_circ)
    circ_recovered = from_pennylane(tape_recovered)

    u_1 = cirq.unitary(base_circ)
    u_2 = cirq.unitary(circ_recovered)

    assert np.allclose(u_1, u_2)
