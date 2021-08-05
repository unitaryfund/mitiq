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

import cirq
import pennylane as qml
from mitiq.interface.mitiq_pennylane import from_pennylane, to_pennylane
from mitiq.utils import _equal


def test_from_pennylane_with_expvals():
    """Tests converting Pennylane expectation values."""
    # <X>
    with qml.tape.QuantumTape() as tape:
        qml.expval(qml.PauliX(wires=[0]))

    circuit = from_pennylane(tape)
    q = cirq.LineQubit(0)
    correct = cirq.Circuit(cirq.H.on(q), cirq.measure(q))
    assert _equal(circuit, correct, require_qubit_equality=False)

    # <Z>
    with qml.tape.QuantumTape() as tape:
        qml.expval(qml.PauliZ(wires=[0]))

    circuit = from_pennylane(tape)
    correct = cirq.Circuit(cirq.measure(q))
    assert _equal(circuit, correct, require_qubit_equality=False)


def test_from_pennylane():
    with qml.tape.QuantumTape() as tape:
        qml.CNOT(wires=[0, 1])

    circuit = from_pennylane(tape)
    correct = cirq.Circuit(cirq.CNOT(*cirq.LineQubit.range(2)))

    # TODO: from_pennylane adds measurements even if there is not measurements
    #  in the tape. This is because tape.to_openqasm(...) measures all qubits
    #  even if there are no measurements in the tape.
    #  Temp patch: Manually remove measurements.
    circuit = circuit[:-1]

    assert _equal(circuit, correct, require_qubit_equality=False)


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
