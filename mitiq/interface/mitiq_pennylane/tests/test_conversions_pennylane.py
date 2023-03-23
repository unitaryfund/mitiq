"""Unit tests for Pennylane <-> Cirq conversions."""
import pytest
import numpy as np
import cirq
import pennylane as qml
from mitiq.interface.mitiq_pennylane import from_pennylane, to_pennylane, UnsupportedQuantumTapeError
from mitiq.utils import _equal

@pytest.mark.order(0)
def test_from_pennylane():
    with qml.tape.QuantumTape() as tape:
        qml.CNOT(wires=[0, 1])
    circuit = from_pennylane(tape)
    correct = cirq.Circuit(cirq.CNOT(*cirq.LineQubit.range(2)))
    assert _equal(circuit, correct, require_qubit_equality=False)

@pytest.mark.order(0)
def test_from_pennylane_unsupported_tapes():
    with qml.tape.QuantumTape() as tape:
        qml.CZ(wires=[0, 'a'])
    with pytest.raises(UnsupportedQuantumTapeError, match='could not sort'):
        from_pennylane(tape)

@pytest.mark.order(0)
def test_no_variance():
    with qml.tape.QuantumTape() as tape:
        qml.CNOT(wires=[0, 1])
        qml.expval(qml.PauliZ(0))
    with pytest.raises(UnsupportedQuantumTapeError, match='Measurements are not supported on the input tape.'):
        from_pennylane(tape)

@pytest.mark.parametrize('random_state', range(10))
def test_to_from_pennylane(random_state):
    circuit = cirq.testing.random_circuit(qubits=4, n_moments=2, op_density=1, random_state=random_state)
    converted = from_pennylane(to_pennylane(circuit))
    cirq.testing.assert_allclose_up_to_global_phase(cirq.unitary(converted), cirq.unitary(circuit), atol=1e-07)

@pytest.mark.order(0)
def test_to_from_pennylane_cnot_same_gates():
    qreg = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(cirq.CNOT(*qreg))
    converted = from_pennylane(to_pennylane(circuit))
    assert _equal(circuit, converted, require_qubit_equality=False)

@pytest.mark.order(0)
def test_to_from_pennylane_identity():
    q = cirq.LineQubit(0)
    circuit = cirq.Circuit()
    converted = from_pennylane(to_pennylane(circuit))
    assert _equal(circuit, converted, require_qubit_equality=False)
    circuit = cirq.Circuit(cirq.I(q))
    converted = from_pennylane(to_pennylane(circuit))
    assert np.allclose(cirq.unitary(circuit), cirq.unitary(converted))

@pytest.mark.order(0)
def test_non_consecutive_wires_error():
    with qml.tape.QuantumTape() as tape:
        qml.CNOT(wires=[0, 2])
    with pytest.raises(UnsupportedQuantumTapeError, match='contiguously pack'):
        from_pennylane(tape)

@pytest.mark.order(0)
def test_integration():
    gates = [qml.PauliX(wires=0), qml.PauliY(wires=0), qml.PauliZ(wires=0), qml.S(wires=0), qml.T(wires=0), qml.RX(0.4, wires=0), qml.RY(0.4, wires=0), qml.RZ(0.4, wires=0), qml.Hadamard(wires=0), qml.Rot(0.4, 0.5, 0.6, wires=1), qml.CRot(0.4, 0.5, 0.6, wires=(0, 1)), qml.Toffoli(wires=(0, 1, 2)), qml.SWAP(wires=(0, 1)), qml.CSWAP(wires=(0, 1, 2)), qml.U1(0.4, wires=0), qml.U2(0.4, 0.5, wires=0), qml.U3(0.4, 0.5, 0.6, wires=0), qml.CRX(0.4, wires=(0, 1)), qml.CRY(0.4, wires=(0, 1)), qml.CRZ(0.4, wires=(0, 1))]
    layers = 3
    np.random.seed(1967)
    gates_per_layers = [np.random.permutation(gates) for _ in range(layers)]
    with qml.tape.QuantumTape() as tape:
        np.random.seed(1967)
        for gates in gates_per_layers:
            for gate in gates:
                qml.apply(gate)
    base_circ = from_pennylane(tape)
    tape_recovered = to_pennylane(base_circ)
    circ_recovered = from_pennylane(tape_recovered)
    u_1 = cirq.unitary(base_circ)
    u_2 = cirq.unitary(circ_recovered)
    cirq.testing.assert_allclose_up_to_global_phase(u_1, u_2, atol=0)