# Copyright (C) Unitary Fund
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

from mitiq.pt.pt import add_paulis, sample_paulis, _generate_lookup_table
import cirq
import pytest
from random import seed

num_qubits = 2
qubits = cirq.LineQubit.range(num_qubits)
circuit = cirq.Circuit()
circuit.append(cirq.CNOT.on_each(zip(qubits, qubits[1:])))


def test_add_paulis():
    twirled = add_paulis(circuit)
    twirled_circuit = cirq.Circuit(
        cirq.X.on(qubits[0]),
        cirq.I.on(qubits[1]),
        cirq.CNOT.on(*qubits),
        cirq.X.on_each(*qubits),
    )

    assert twirled == twirled_circuit


def test_generate_lookup_table():
    def _test_generate_lookup_table(gate, expected_length):
        assert len(_generate_lookup_table(gate)) == expected_length

    def _test_generate_lookup_table_exception(gate):
        with pytest.raises(ValueError):
            _generate_lookup_table(gate)

    _test_generate_lookup_table('CNOT', 16)
    _test_generate_lookup_table('CZ', 16)
    _test_generate_lookup_table_exception('INVALID_GATE')

@pytest.mark.parametrize("gate, seed_val, expected_tuple", [
    ('CNOT', 0, (cirq.Y, cirq.X, cirq.X, cirq.Y)),
    ('CZ', 0, (cirq.Y, cirq.Z, cirq.Z, cirq.Y)),
    ('CNOT', 1, (cirq.I, cirq.X, cirq.X, cirq.I)),
    ('CZ', 1, (cirq.Z, cirq.I, cirq.I, cirq.Z))
])
def test_sample_paulis(gate, seed_val, expected_tuple):
    seed(seed_val)  # Fix random seed for reproducibility
    P1, P2, R1, R2 = sample_paulis(gate)
    assert (P1, P2, R1, R2) == expected_tuple

def test_sample_paulis():
    with pytest.raises(ValueError):
        sample_paulis('INVALID_GATE')
