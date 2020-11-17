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

"""Unit tests for scaling noise by unitary folding."""
from copy import deepcopy

import numpy as np
import pytest
from cirq import (
    Circuit,
    GridQubit,
    LineQubit,
    ops,
    inverse,
    equal_up_to_global_phase,
    InsertStrategy,
    testing,
)
from cirq.google import Sycamore
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.quantum_info.operators import Operator
from pyquil import Program
from pyquil.quilbase import Pragma

from mitiq.utils import _equal
from mitiq.conversions import (
    CircuitConversionError,
    convert_to_mitiq,
    convert_from_mitiq,
)
from mitiq.zne.scaling.folding import (
    UnfoldableGateError,
    UnfoldableCircuitError,
    _is_measurement,
    _pop_measurements,
    _append_measurements,
    squash_moments,
    _update_moment_indices,
    _default_weight,
    _compute_weight,
    _fold_gate_at_index_in_moment,
    _fold_gates_in_moment,
    _fold_gates,
    _fold_moments,
    fold_gates_from_left,
    fold_gates_from_right,
    fold_gates_at_random,
    _fold_local,
    fold_global,
)


def test_is_measurement():
    """Tests for checking if operations are measurements."""
    # Test circuit:
    # 0: ───H───X───Z───
    qbit = LineQubit(0)
    circ = Circuit(
        [ops.H.on(qbit), ops.X.on(qbit), ops.Z.on(qbit), ops.measure(qbit)]
    )
    for (i, op) in enumerate(circ.all_operations()):
        if i == 3:
            assert _is_measurement(op)
        else:
            assert not _is_measurement(op)


def test_pop_measurements_and_add_measurements():
    """Tests popping measurements from a circuit.."""
    # Test circuit:
    # 0: ───H───T───@───M───
    #               │   │
    # 1: ───H───M───┼───┼───
    #               │   │
    # 2: ───H───────X───M───
    qreg = LineQubit.range(3)
    circ = Circuit(
        [ops.H.on_each(qreg)],
        [ops.T.on(qreg[0])],
        [ops.measure(qreg[1])],
        [ops.CNOT.on(qreg[0], qreg[2])],
        [ops.measure(qreg[0], qreg[2])],
    )
    copy = deepcopy(circ)
    measurements = _pop_measurements(copy)
    correct = Circuit(
        [ops.H.on_each(qreg)],
        [ops.T.on(qreg[0])],
        [ops.CNOT.on(qreg[0], qreg[2])],
    )
    assert _equal(copy, correct)
    _append_measurements(copy, measurements)
    assert _equal(copy, circ)


def test_squash_moments_two_qubits():
    """Tests squashing moments in a two-qubit circuit with 'staggered' single
    qubit gats.
    """
    # Test circuit:
    # 0: ───────H───────H───────H───────H───────H───
    #
    # 1: ───H───────H───────H───────H───────H───────

    # Get the test circuit
    d = 10
    qreg = LineQubit.range(2)
    circuit = Circuit()
    for i in range(d):
        circuit.insert(0, ops.H.on(qreg[i % 2]), strategy=InsertStrategy.NEW)
    assert len(circuit) == d

    # Squash the moments
    squashed = squash_moments(circuit)
    assert len(squashed) == d // 2


def test_squash_moments_returns_new_circuit_and_doesnt_modify_input_circuit():
    """Tests that squash moments returns a new circuit and doesn't modify the
    input circuit.
    """
    qbit = GridQubit(0, 0)
    circ = Circuit(ops.H.on(qbit))
    squashed = squash_moments(circ)
    assert len(squashed) == 1
    assert circ is not squashed
    assert _equal(circ, Circuit(ops.H.on(qbit)))


def test_squash_moments_retains_device():
    """Tests that the returned circuit from squash_moments has the same device
    as the input circuit.
    """
    circuit = Circuit(device=Sycamore)
    squashed = squash_moments(circuit)
    assert squashed.device == Sycamore


def test_squash_moments_never_increases_moments():
    """Squashes moments for several random circuits and ensures the squashed
    circuit always <= # moments as the input circuit.
    """
    for _ in range(50):
        circuit = testing.random_circuit(
            qubits=5, n_moments=8, op_density=0.75
        )
        squashed = squash_moments(circuit)
        assert len(squashed) <= len(circuit)


def test_update_moment_indices():
    """Tests indices of moments are properly updated."""
    moment_indices = {i: i for i in range(5)}
    _update_moment_indices(moment_indices, 3)
    assert moment_indices == {0: 0, 1: 1, 2: 2, 3: 5, 4: 6}
    _update_moment_indices(moment_indices, 0)
    assert moment_indices == {0: 2, 1: 3, 2: 4, 3: 7, 4: 8}
    _update_moment_indices(moment_indices, 4)
    assert moment_indices == {0: 2, 1: 3, 2: 4, 3: 7, 4: 10}
    with pytest.raises(ValueError):
        _update_moment_indices(moment_indices, 6)


def test_fold_gate_at_index_in_moment_one_qubit():
    """Tests local folding with a moment, index for a one qubit circuit."""
    # Test circuit:
    # 0: ───H───X───Z───
    qbit = LineQubit(0)
    circ = Circuit([ops.H.on(qbit), ops.X.on(qbit), ops.Z.on(qbit)])
    # Fold the zeroth operation in the zeroth moment
    folded = deepcopy(circ)
    _fold_gate_at_index_in_moment(folded, moment_index=0, gate_index=0)
    assert folded == Circuit(
        [ops.H.on(qbit)] * 3 + [ops.X.on(qbit)] + [ops.Z.on(qbit)]
    )
    # Fold the zeroth operation in the first moment
    folded = deepcopy(circ)
    _fold_gate_at_index_in_moment(folded, moment_index=1, gate_index=0)
    assert folded == Circuit(
        [ops.H.on(qbit)] + [ops.X.on(qbit)] * 3 + [ops.Z.on(qbit)]
    )
    # Fold the zeroth operation in the second moment
    folded = deepcopy(circ)
    _fold_gate_at_index_in_moment(folded, moment_index=2, gate_index=0)
    assert folded == Circuit(
        [ops.H.on(qbit)] + [ops.X.on(qbit)] + [ops.Z.on(qbit)] * 3
    )
    # Make sure the original circuit wasn't modified
    old = Circuit([ops.H.on(qbit), ops.X.on(qbit), ops.Z.on(qbit)])
    assert _equal(circ, old)


def test_fold_gate_at_index_in_moment_two_qubits():
    """Tests local folding with a moment, index for a two qubit circuit with
    single qubit gates.
    """
    # Test circuit:
    # 0: ───H───T───
    #
    # 1: ───H───T───
    qreg = LineQubit.range(2)
    circ = Circuit([ops.H.on_each(*qreg), ops.T.on_each(*qreg)])

    # Fold the zeroth operation in the zeroth moment
    folded = deepcopy(circ)
    _fold_gate_at_index_in_moment(folded, moment_index=0, gate_index=0)
    correct = Circuit(
        [ops.H.on(qreg[0]), ops.H.on(qreg[0]) ** -1]
        + list(circ.all_operations())
    )
    assert _equal(folded, correct)

    # Fold the first operation in the zeroth moment
    folded = deepcopy(circ)
    _fold_gate_at_index_in_moment(folded, moment_index=0, gate_index=1)
    correct = Circuit(
        [ops.H.on(qreg[1]), ops.H.on(qreg[1]) ** -1]
        + list(circ.all_operations())
    )
    assert _equal(folded, correct)

    # Fold the zeroth operation in the first moment
    folded = deepcopy(circ)
    _fold_gate_at_index_in_moment(folded, moment_index=1, gate_index=0)
    correct = Circuit(
        [ops.H.on_each(*qreg)],
        [ops.T.on(qreg[0]), ops.T.on(qreg[0]) ** -1, ops.T.on(qreg[0])],
        [ops.T.on(qreg[1])],
    )
    assert _equal(folded, correct)

    # Fold the first operation in the first moment
    folded = deepcopy(circ)
    _fold_gate_at_index_in_moment(folded, moment_index=1, gate_index=1)
    correct = Circuit(
        [ops.H.on_each(*qreg)],
        [ops.T.on(qreg[1]), ops.T.on(qreg[1]) ** -1, ops.T.on(qreg[1])],
        [ops.T.on(qreg[0])],
    )
    assert _equal(folded, correct)

    # Make sure the original circuit wasn't modified
    assert Circuit([ops.H.on_each(*qreg), ops.T.on_each(*qreg)])


def test_fold_gate_at_index_in_moment_two_qubit_gates():
    """Tests local folding with a moment, index for a two qubit circuit with
    two qubit gates.
    """
    # Test circuit:
    # 0: ───H───@───
    #           │
    # 1: ───────X───
    qreg = LineQubit.range(2)
    circ = Circuit([ops.H.on(qreg[0]), ops.CNOT.on(*qreg)])

    # Fold the zeroth operation in the zeroth moment
    folded = deepcopy(circ)
    _fold_gate_at_index_in_moment(folded, moment_index=0, gate_index=0)
    correct = Circuit(
        [ops.H.on(qreg[0]) ** -1, ops.H.on(qreg[0])]
        + list(circ.all_operations())
    )
    assert _equal(folded, correct)

    # Fold the zeroth operation in the first moment
    folded = deepcopy(circ)
    _fold_gate_at_index_in_moment(folded, moment_index=1, gate_index=0)
    correct = Circuit(list(circ.all_operations()) + [ops.CNOT.on(*qreg)] * 2)
    assert _equal(folded, correct)

    # Make sure the original circuit wasn't modified
    old = Circuit([ops.H.on(qreg[0]), ops.CNOT.on(*qreg)])
    assert _equal(circ, old)


def test_fold_gate_at_index_in_moment_empty_circuit():
    """Tests local folding with a moment, index with an empty circuit."""
    circ = Circuit()

    # Fold the zeroth operation in the zeroth moment
    with pytest.raises(IndexError):
        _fold_gate_at_index_in_moment(circ, moment_index=0, gate_index=0)


def test_fold_gate_at_index_in_moment_bad_moment():
    """Tests local folding with a moment index not in the input circuit."""
    qreg = GridQubit.rect(2, 2)
    circ = Circuit([ops.H.on_each(*qreg)])
    with pytest.raises(IndexError):
        _fold_gate_at_index_in_moment(circ, 1, 0)


def test_unfoldable_gate_error_cirq():
    """Tries to fold a gate without an inverse and checks an error is
    raised."""
    qbit = LineQubit(0)
    circ = Circuit(ops.measure(qbit))
    with pytest.raises(UnfoldableGateError):
        _fold_gate_at_index_in_moment(circ, moment_index=0, gate_index=0)


def test_unfoldable_gate_error_qiskit():
    """Tries to fold a gate without an inverse and checks an error is
    raised."""
    qreg = QuantumRegister(1)
    creg = ClassicalRegister(1)
    circ = QuantumCircuit(qreg, creg)
    circ.measure(qreg, creg)
    cirq_circuit, _ = convert_to_mitiq(circ)
    with pytest.raises(UnfoldableGateError):
        _fold_gate_at_index_in_moment(
            cirq_circuit, moment_index=0, gate_index=0
        )


def test_fold_gates_in_moment_single_qubit_gates():
    """Tests folding gates at given indices within a moment."""
    # Test circuit:
    # 0: ───H───T───
    #
    # 1: ───H───T───
    #
    # 2: ───H───T───
    qreg = LineQubit.range(3)
    circ = Circuit([ops.H.on_each(*qreg), ops.T.on_each(*qreg)])

    # Fold all gates in the zeroth moment
    folded = deepcopy(circ)
    _fold_gates_in_moment(folded, moment_index=0, gate_indices=[0, 1, 2])
    correct = Circuit([ops.H.on_each(*qreg)] * 3 + [ops.T.on_each(*qreg)])
    assert _equal(folded, correct)

    # Fold a subset of gates in the first moment
    folded = deepcopy(circ)
    _fold_gates_in_moment(folded, moment_index=1, gate_indices=[0, 2])
    correct = Circuit(
        [ops.H.on_each(*qreg)],
        [ops.T.on(qreg[0]), ops.T.on(qreg[0]) ** -1],
        [ops.T.on(qreg[2]), ops.T.on(qreg[2]) ** -1],
        [ops.T.on_each(*qreg)],
    )
    assert _equal(folded, correct)


def test_fold_gates_in_moment_multi_qubit_gates():
    """Tests folding gates at given indices within a moment."""
    # Test circuit:
    # 0: ───H───@───@───
    #           │   │
    # 1: ───H───X───@───
    #               │
    # 2: ───H───T───X───
    qreg = LineQubit.range(3)
    circ = Circuit(
        [
            ops.H.on_each(*qreg),
            ops.CNOT.on(qreg[0], qreg[1]),
            ops.T.on(qreg[2]),
            ops.TOFFOLI.on(*qreg),
        ]
    )

    # Fold the CNOT gate in the first moment
    folded = deepcopy(circ)
    _fold_gates_in_moment(folded, moment_index=1, gate_indices=[0])
    correct = Circuit(
        [ops.H.on_each(*qreg)],
        [ops.CNOT.on(qreg[0], qreg[1])] * 3,
        [ops.T.on(qreg[2]), ops.TOFFOLI.on(*qreg)],
    )
    assert _equal(folded, correct)

    # Fold the T gate in the first moment
    folded = deepcopy(circ)
    _fold_gates_in_moment(folded, moment_index=1, gate_indices=[0, 1])
    correct = Circuit(
        [ops.H.on_each(*qreg)],
        [ops.CNOT.on(qreg[0], qreg[1])] * 3,
        [ops.T.on(qreg[2]), ops.T.on(qreg[2]) ** -1, ops.T.on(qreg[2])],
        [ops.TOFFOLI.on(*qreg)],
    )
    assert _equal(folded, correct)

    # Fold the Toffoli gate in the second moment
    _fold_gates_in_moment(circ, moment_index=2, gate_indices=[0])
    correct = Circuit(
        [ops.H.on_each(*qreg)] + [ops.CNOT.on(qreg[0], qreg[1])],
        [ops.T.on(qreg[2])],
        [ops.TOFFOLI.on(*qreg)] * 3,
    )
    assert _equal(circ, correct)


def test_fold_gates():
    """Test folding gates at specified indices within specified moments."""
    # Test circuit:
    # 0: ───H───@───@───
    #           │   │
    # 1: ───H───X───@───
    #               │
    # 2: ───H───T───X───
    qreg = LineQubit.range(3)
    circ = Circuit(
        [
            ops.H.on_each(*qreg),
            ops.CNOT.on(qreg[0], qreg[1]),
            ops.T.on(qreg[2]),
            ops.TOFFOLI.on(*qreg),
        ]
    )
    folded = _fold_gates(
        circ, moment_indices=[0, 1], gate_indices=[[0, 1, 2], [1]]
    )
    correct = Circuit(
        [ops.H.on_each(*qreg)] * 3,
        [ops.CNOT.on(qreg[0], qreg[1])],
        [ops.T.on(qreg[2]), ops.T.on(qreg[2]) ** -1, ops.T.on(qreg[2])],
        [ops.TOFFOLI.on(*qreg)],
    )
    assert _equal(folded, correct)


def test_fold_moments():
    """Tests folding moments in a circuit."""
    # Test circuit
    # 0: ───H───@───@───
    #           │   │
    # 1: ───H───X───@───
    #               │
    # 2: ───H───T───X───
    qreg = LineQubit.range(3)
    base = Circuit(
        [
            ops.H.on_each(*qreg),
            ops.CNOT.on(qreg[0], qreg[1]),
            ops.T.on(qreg[2]),
            ops.TOFFOLI.on(*qreg),
        ]
    )

    # Fold a single moment
    circ = deepcopy(base)
    _fold_moments(circ, moment_indices=[0])
    correct = Circuit([ops.H.on_each(*qreg)] * 2, list(base.all_operations()))
    assert _equal(circ, correct)

    # Fold another single moment
    circ = deepcopy(base)
    _fold_moments(circ, moment_indices=[2])
    correct = Circuit(list(base.all_operations()), [ops.TOFFOLI.on(*qreg)] * 2)
    assert _equal(circ, correct)

    # Fold two moments
    circ = deepcopy(base)
    _fold_moments(circ, moment_indices=[0, 2])
    correct = Circuit(
        [ops.H.on_each(*qreg)] * 2,
        list(base.all_operations()),
        [ops.TOFFOLI.on(*qreg)] * 2,
    )
    assert _equal(circ, correct)

    # Fold three moments
    circ = deepcopy(base)
    _fold_moments(circ, moment_indices=[0, 1, 2])
    correct = Circuit(
        [ops.H.on_each(*qreg)] * 3,
        [ops.CNOT.on(qreg[0], qreg[1])] * 3,
        [ops.T.on(qreg[2]), ops.T.on(qreg[2]) ** -1, ops.T.on(qreg[2])],
        [ops.TOFFOLI.on(*qreg)] * 3,
    )
    assert _equal(circ, correct)


def test_fold_from_left_two_qubits():
    qreg = LineQubit.range(2)
    circ = Circuit(
        [
            ops.H.on_each(*qreg),
            ops.CNOT.on(qreg[0], qreg[1]),
            ops.T.on(qreg[1]),
        ]
    )

    # Intermediate scale factor
    folded = fold_gates_from_left(circ, scale_factor=2.5)
    correct = Circuit(
        [ops.H.on_each(*qreg)] * 3,
        [ops.CNOT.on(*qreg)] * 3,
        [ops.T.on(qreg[1])],
    )
    assert _equal(folded, correct)

    # Full scale factor
    folded = fold_gates_from_left(circ, scale_factor=3)
    correct = Circuit(
        [ops.H.on_each(*qreg)] * 3,
        [ops.CNOT.on(*qreg)] * 3,
        [ops.T.on(qreg[1]), ops.T.on(qreg[1]) ** -1, ops.T.on(qreg[1])],
    )
    assert _equal(folded, correct)


def test_fold_from_left_three_qubits():
    """Unit test for folding gates from left to scale a circuit."""
    qreg = LineQubit.range(3)
    circ = Circuit(
        [
            ops.H.on_each(*qreg),
            ops.CNOT.on(qreg[0], qreg[1]),
            ops.T.on(qreg[2]),
            ops.TOFFOLI.on(*qreg),
        ]
    )

    folded = fold_gates_from_left(circ, scale_factor=2)
    correct = Circuit([ops.H.on_each(*qreg)] * 2, list(circ.all_operations()))
    assert _equal(folded, correct)


def test_fold_from_left_no_stretch():
    """Unit test for folding gates from left for a scale factor of one."""
    circuit = testing.random_circuit(qubits=2, n_moments=10, op_density=0.99)
    folded = fold_gates_from_left(circuit, scale_factor=1)
    assert _equal(folded, circuit)
    assert not (folded is circuit)


def test_fold_from_left_scale_factor_larger_than_three():
    """Tests folding from left with a scale_factor larger than three."""
    qreg = LineQubit.range(2)
    circuit = Circuit([ops.SWAP.on(*qreg)], [ops.CNOT.on(*qreg)])
    folded = fold_gates_from_left(circuit, scale_factor=6.0)
    correct = Circuit([ops.SWAP.on(*qreg)] * 9, [ops.CNOT.on(*qreg)] * 3)
    assert _equal(folded, correct)


def test_fold_from_right_scale_factor_larger_than_three():
    """Tests folding from right with a scale_factor larger than three."""
    qreg = LineQubit.range(2)
    circuit = Circuit([ops.SWAP.on(*qreg)], [ops.CNOT.on(*qreg)])
    folded = fold_gates_from_right(circuit, scale_factor=6.0)
    correct = Circuit([ops.SWAP.on(*qreg)] * 3, [ops.CNOT.on(*qreg)] * 9)
    assert _equal(folded, correct)


def test_fold_from_left_with_terminal_measurements_min_stretch():
    """Tests folding from left with terminal measurements."""
    # Test circuit
    # 0: ───H───@───@───M───
    #           │   │
    # 1: ───H───X───@───M───
    #               │
    # 2: ───H───T───X───M───
    qreg = LineQubit.range(3)
    circ = Circuit(
        [ops.H.on_each(*qreg)],
        [ops.CNOT.on(qreg[0], qreg[1])],
        [ops.T.on(qreg[2])],
        [ops.TOFFOLI.on(*qreg)],
        [ops.measure_each(*qreg)],
    )
    folded = fold_gates_from_left(circ, scale_factor=1.0)
    correct = Circuit(
        [ops.H.on_each(*qreg)],
        [ops.CNOT.on(qreg[0], qreg[1])],
        [ops.T.on(qreg[2])],
        [ops.TOFFOLI.on(*qreg)],
        [ops.measure_each(*qreg)],
    )
    assert _equal(folded, correct)


def test_fold_from_left_with_terminal_measurements_max_stretch():
    """Tests folding from left with terminal measurements."""
    # Test circuit
    # 0: ───H───@───@───M───
    #           │   │
    # 1: ───H───X───@───M───
    #               │
    # 2: ───H───T───X───M───
    qreg = LineQubit.range(3)
    circ = Circuit(
        [ops.H.on_each(*qreg)],
        [ops.CNOT.on(qreg[0], qreg[1])],
        [ops.T.on(qreg[2])],
        [ops.TOFFOLI.on(*qreg)],
        [ops.measure_each(*qreg)],
    )
    folded = fold_gates_from_left(circ, scale_factor=3.0)
    correct = Circuit(
        [ops.H.on_each(*qreg)] * 3,
        [ops.CNOT.on(qreg[0], qreg[1])] * 3,
        [ops.T.on(qreg[2]), ops.T.on(qreg[2]) ** -1, ops.T.on(qreg[2])],
        [ops.TOFFOLI.on(*qreg)] * 3,
        [ops.measure_each(*qreg)],
    )
    assert _equal(folded, correct)


@pytest.mark.parametrize(
    "fold_method",
    [
        fold_gates_from_left,
        fold_gates_from_right,
        fold_gates_at_random,
        fold_global,
    ],
)
def test_fold_with_intermediate_measurements_raises_error(fold_method):
    """Tests local folding functions raise an error on circuits with
    intermediate measurements.
    """
    qbit = LineQubit(0)
    circ = Circuit([ops.H.on(qbit)], [ops.measure(qbit)], [ops.T.on(qbit)])
    with pytest.raises(
        UnfoldableCircuitError,
        match="Circuit contains intermediate measurements",
    ):
        fold_method(circ, scale_factor=3.0)


@pytest.mark.parametrize(
    "fold_method",
    [
        fold_gates_from_left,
        fold_gates_from_right,
        fold_gates_at_random,
        fold_global,
    ],
)
def test_fold_with_channels_raises_error(fold_method):
    """Tests local folding functions raise an error on circuits with
    non-unitary channels (which are not measurements).
    """
    qbit = LineQubit(0)
    circ = Circuit(
        ops.H.on(qbit), ops.depolarize(p=0.1).on(qbit), ops.measure(qbit)
    )
    with pytest.raises(
        UnfoldableCircuitError, match="Circuit contains non-unitary channels"
    ):
        fold_method(circ, scale_factor=3.0)


def test_fold_from_right_basic():
    """Tests folding gates from the right for a two-qubit circuit."""
    # Test circuit:
    # 0: ───H───@───────
    #           │
    # 1: ───H───X───T───
    qreg = LineQubit.range(2)
    circ = Circuit(
        [ops.H.on_each(*qreg)],
        [ops.CNOT.on(qreg[0], qreg[1])],
        [ops.T.on(qreg[1])],
    )

    # Small scale factor
    folded = fold_gates_from_right(circ, scale_factor=1.5)
    correct = Circuit(
        [ops.H.on_each(*qreg)],
        [ops.CNOT.on(*qreg)],
        [ops.T.on(qreg[1]), ops.T.on(qreg[1]) ** -1, ops.T.on(qreg[1])],
    )
    assert _equal(folded, correct)

    # Intermediate scale factor
    folded = fold_gates_from_right(circ, scale_factor=2.5)
    correct = Circuit(
        [ops.H.on(qreg[0])] * 3,
        [ops.H.on(qreg[1])],
        [ops.CNOT.on(*qreg)] * 3,
        [ops.T.on(qreg[1]), ops.T.on(qreg[1]) ** -1, ops.T.on(qreg[1])],
    )
    assert _equal(folded, correct)


def test_fold_from_right_max_stretch():
    """Tests that fold from right = fold from left with maximum scale."""
    # Test circuit
    # 0: ───H───@───@───
    #           │   │
    # 1: ───H───X───@───
    #               │
    # 2: ───H───T───X───
    qreg = LineQubit.range(3)
    circ = Circuit(
        [ops.H.on_each(*qreg)],
        [ops.CNOT.on(qreg[0], qreg[1])],
        [ops.T.on(qreg[2])],
        [ops.TOFFOLI.on(*qreg)],
    )

    left_folded = fold_gates_from_left(circ, scale_factor=3.0)
    right_folded = fold_gates_from_right(circ, scale_factor=3.0)
    assert _equal(left_folded, right_folded)


def test_fold_from_right_with_terminal_measurements_min_stretch():
    """Tests folding from left with terminal measurements."""
    # Test circuit
    # 0: ───H───@───@───M───
    #           │   │
    # 1: ───H───X───@───M───
    #               │
    # 2: ───H───T───X───M───
    qreg = LineQubit.range(3)
    circ = Circuit(
        [ops.H.on_each(*qreg)],
        [ops.CNOT.on(qreg[0], qreg[1])],
        [ops.T.on(qreg[2])],
        [ops.TOFFOLI.on(*qreg)],
        [ops.measure_each(*qreg)],
    )
    folded = fold_gates_from_right(circ, scale_factor=1.0)
    correct = Circuit(
        [ops.H.on_each(*qreg)],
        [ops.CNOT.on(qreg[0], qreg[1])],
        [ops.T.on(qreg[2])],
        [ops.TOFFOLI.on(*qreg)],
        [ops.measure_each(*qreg)],
    )
    assert _equal(folded, correct)
    assert _equal(circ, correct)


def test_fold_from_right_with_terminal_measurements_max_stretch():
    """Tests folding from left with terminal measurements."""
    # Test circuit
    # 0: ───H───@───@───M───
    #           │   │
    # 1: ───H───X───@───M───
    #               │
    # 2: ───H───T───X───M───
    qreg = LineQubit.range(3)
    circ = Circuit(
        [ops.H.on_each(*qreg)],
        [ops.CNOT.on(qreg[0], qreg[1])],
        [ops.T.on(qreg[2])],
        [ops.TOFFOLI.on(*qreg)],
        [ops.measure_each(*qreg)],
    )
    folded = fold_gates_from_right(circ, scale_factor=3.0)
    correct = Circuit(
        [ops.H.on_each(*qreg)] * 3,
        [ops.CNOT.on(qreg[0], qreg[1])] * 3,
        [ops.T.on(qreg[2]), ops.T.on(qreg[2]) ** -1, ops.T.on(qreg[2])],
        [ops.TOFFOLI.on(*qreg)] * 3,
        [ops.measure_each(*qreg)],
    )
    assert _equal(folded, correct)

    # Make sure the original circuit is not modified
    original = Circuit(
        [ops.H.on_each(*qreg)],
        [ops.CNOT.on(qreg[0], qreg[1])],
        [ops.T.on(qreg[2])],
        [ops.TOFFOLI.on(*qreg)],
        [ops.measure_each(*qreg)],
    )
    assert _equal(circ, original)


def test_fold_right_retains_terminal_measurements_in_input_circuit():
    """Tests that folding from the right doesn't modify the terminal
    measurements in the input circuit.
    """
    qbit = LineQubit(1)
    circ = Circuit(ops.H.on(qbit), ops.measure(qbit))
    folded = fold_gates_from_right(circ, scale_factor=1.0)
    assert _equal(circ, folded)


def test_fold_gates_at_random_no_stretch():
    """Tests folded circuit is identical for a scale factor of one."""
    circuit = testing.random_circuit(qubits=3, n_moments=10, op_density=0.99)
    folded = fold_gates_at_random(circuit, scale_factor=1, seed=None)
    assert _equal(folded, circuit)


def test_fold_gates_at_random_seed_one_qubit():
    """Test for folding gates at random on a one qubit circuit with a seed for
    repeated behavior.
    """
    qubit = LineQubit(0)
    circuit = Circuit([ops.X.on(qubit), ops.Y.on(qubit), ops.Z.on(qubit)])
    # Small scale
    folded = fold_gates_at_random(circuit, scale_factor=1.4, seed=1)
    correct = Circuit(
        [ops.X.on(qubit)], [ops.Y.on(qubit)] * 3, [ops.Z.on(qubit)]
    )
    assert _equal(folded, correct)

    # Medium scale, fold two gates
    folded = fold_gates_at_random(circuit, scale_factor=2.5, seed=1)
    correct = Circuit(
        [ops.X.on(qubit)], [ops.Y.on(qubit)] * 3, [ops.Z.on(qubit)] * 3,
    )
    assert _equal(folded, correct)

    # Max scale, fold three gates
    folded = fold_gates_at_random(circuit, scale_factor=3, seed=1)
    correct = Circuit(
        [ops.X.on(qubit)] * 3, [ops.Y.on(qubit)] * 3, [ops.Z.on(qubit)] * 3,
    )
    assert _equal(folded, correct)


def test_fold_random_min_stretch():
    """Tests that folding at random with min scale returns a copy of the
    input circuit.
    """
    # Test circuit
    # 0: ───H───@───@───
    #           │   │
    # 1: ───H───X───@───
    #               │
    # 2: ───H───T───X───
    qreg = LineQubit.range(3)
    circ = Circuit(
        [ops.H.on_each(*qreg)],
        [ops.CNOT.on(qreg[0], qreg[1])],
        [ops.T.on(qreg[2])],
        [ops.TOFFOLI.on(*qreg)],
    )

    folded = fold_gates_at_random(circ, scale_factor=1, seed=1)
    assert _equal(folded, circ)
    assert folded is not circ


def test_fold_random_max_stretch():
    """Tests that folding at random with max scale folds all gates on a
    multi-qubit circuit.
    """
    # Test circuit
    # 0: ───H───@───@───
    #           │   │
    # 1: ───H───X───@───
    #               │
    # 2: ───H───T───X───
    qreg = LineQubit.range(3)
    circ = Circuit(
        [ops.H.on_each(*qreg)],
        [ops.CNOT.on(qreg[0], qreg[1])],
        [ops.T.on(qreg[2])],
        [ops.TOFFOLI.on(*qreg)],
    )

    folded = fold_gates_at_random(circ, scale_factor=3, seed=1)
    correct = Circuit(
        [ops.H.on_each(*qreg)] * 3,
        [ops.CNOT.on(qreg[0], qreg[1])] * 3,
        [ops.T.on(qreg[2]), ops.T.on(qreg[2]) ** -1, ops.T.on(qreg[2])],
        [ops.TOFFOLI.on(*qreg)] * 3,
    )
    assert _equal(folded, correct)


def test_fold_random_scale_factor_larger_than_three():
    """Folds at random with a scale_factor larger than three."""
    qreg = LineQubit.range(2)
    circuit = Circuit([ops.SWAP.on(*qreg)], [ops.CNOT.on(*qreg)])
    folded = fold_gates_at_random(circuit, scale_factor=6.0, seed=1)
    correct = Circuit([ops.SWAP.on(*qreg)] * 5, [ops.CNOT.on(*qreg)] * 7)
    assert len(folded) == 12
    assert _equal(folded, correct)


def test_fold_random_no_repeats():
    """Tests folding at random to ensure that no gates are folded twice and
    folded gates are not folded again.
    """
    # Test circuit:
    # 0: ───H───@───Y───@───
    #           │       │
    # 1: ───────X───X───@───
    # Note that each gate only occurs once and is self-inverse.
    # This allows us to check that no gates are folded more than once
    qreg = LineQubit.range(2)
    circ = Circuit(
        [ops.H.on_each(qreg[0])],
        [ops.CNOT.on(*qreg)],
        [ops.X.on(qreg[1])],
        [ops.Y.on(qreg[0])],
        [ops.CZ.on(*qreg)],
    )
    circuit_ops = set(circ.all_operations())

    for scale in np.linspace(1.0, 3.0, 5):
        folded = fold_gates_at_random(circ, scale_factor=scale, seed=1)
        gates = list(folded.all_operations())
        counts = {gate: gates.count(gate) for gate in circuit_ops}
        assert all(count <= 3 for count in counts.values())


def test_fold_random_with_terminal_measurements_min_stretch():
    """Tests folding from left with terminal measurements."""
    # Test circuit
    # 0: ───H───@───@───M───
    #           │   │
    # 1: ───H───X───@───M───
    #               │
    # 2: ───H───T───X───M───
    qreg = LineQubit.range(3)
    circ = Circuit(
        [ops.H.on_each(*qreg)],
        [ops.CNOT.on(qreg[0], qreg[1])],
        [ops.T.on(qreg[2])],
        [ops.TOFFOLI.on(*qreg)],
        [ops.measure_each(*qreg)],
    )
    folded = fold_gates_at_random(circ, scale_factor=1.0)
    correct = Circuit(
        [ops.H.on_each(*qreg)],
        [ops.CNOT.on(qreg[0], qreg[1])],
        [ops.T.on(qreg[2])],
        [ops.TOFFOLI.on(*qreg)],
        [ops.measure_each(*qreg)],
    )
    assert _equal(folded, correct)


def test_fold_random_with_terminal_measurements_max_stretch():
    """Tests folding from left with terminal measurements."""
    # Test circuit
    # 0: ───H───@───@───M───
    #           │   │
    # 1: ───H───X───@───M───
    #               │
    # 2: ───H───T───X───M───
    qreg = LineQubit.range(3)
    circ = Circuit(
        [ops.H.on_each(*qreg)],
        [ops.CNOT.on(qreg[0], qreg[1])],
        [ops.T.on(qreg[2])],
        [ops.TOFFOLI.on(*qreg)],
        [ops.measure_each(*qreg)],
    )
    folded = fold_gates_at_random(circ, scale_factor=3.0)
    correct = Circuit(
        [ops.H.on_each(*qreg)] * 3,
        [ops.CNOT.on(qreg[0], qreg[1])] * 3,
        [ops.T.on(qreg[2]), ops.T.on(qreg[2]) ** -1, ops.T.on(qreg[2])],
        [ops.TOFFOLI.on(*qreg)] * 3,
        [ops.measure_each(*qreg)],
    )
    assert _equal(folded, correct)


def test_fold_local_small_stretch_from_left():
    """Test for local folding with scale < 3."""
    qreg = LineQubit.range(3)
    circ = Circuit(
        [
            ops.H.on_each(*qreg),
            ops.CNOT.on(qreg[0], qreg[1]),
            ops.T.on(qreg[2]),
            ops.TOFFOLI.on(*qreg),
        ]
    )

    folded = _fold_local(
        circ, scale_factor=2, fold_method=fold_gates_from_left
    )
    correct = Circuit([ops.H.on_each(*qreg)] * 2, list(circ.all_operations()))
    assert _equal(folded, correct)


def test_fold_local_stretch_three_from_left():
    """Test for local folding with scale > 3."""
    qreg = LineQubit.range(3)
    circ = Circuit(
        [
            ops.H.on_each(*qreg),
            ops.CNOT.on(qreg[0], qreg[1]),
            ops.T.on(qreg[2]),
            ops.TOFFOLI.on(*qreg),
        ]
    )
    folded = _fold_local(
        circ, scale_factor=3, fold_method=fold_gates_from_left
    )
    assert _equal(folded, fold_gates_from_left(circ, scale_factor=3))


@pytest.mark.parametrize("squash", [True, False])
def test_fold_local_big_stretch_from_left(squash: bool):
    """Test for local folding with scale > 3."""
    qreg = LineQubit.range(3)
    circ = Circuit(
        [
            ops.H.on_each(*qreg),
            ops.CNOT.on(qreg[0], qreg[1]),
            ops.T.on(qreg[2]),
            ops.TOFFOLI.on(*qreg),
        ]
    )
    folded = _fold_local(
        circ,
        scale_factor=4,
        fold_method=fold_gates_from_left,
        squash_moments=squash,
    )
    correct = Circuit(
        [ops.H.on(qreg[0])] * 7,
        [ops.H.on(qreg[1])] * 5,
        [ops.H.on(qreg[2])] * 3,
        [ops.CNOT.on(qreg[0], qreg[1])] * 3,
        [ops.T.on(qreg[2]), ops.T.on(qreg[2]) ** -1, ops.T.on(qreg[2])],
        [ops.TOFFOLI.on(*qreg)] * 3,
    )
    assert _equal(folded, correct)


def test_global_fold_min_stretch():
    """Tests that global fold with scale = 1 is the same circuit."""
    # Test circuit
    # 0: ───H───@───@───
    #           │   │
    # 1: ───H───X───@───
    #               │
    # 2: ───H───T───X───
    qreg = LineQubit.range(3)
    circ = Circuit(
        [ops.H.on_each(*qreg)],
        [ops.CNOT.on(qreg[0], qreg[1])],
        [ops.T.on(qreg[2])],
        [ops.TOFFOLI.on(*qreg)],
    )

    folded = fold_global(circ, 1.0)
    assert _equal(folded, circ)
    assert folded is not circ


def test_global_fold_min_stretch_with_terminal_measurements():
    """Tests that global fold with scale = 1 is the same circuit."""
    # Test circuit
    # 0: ───H───@───@───M───
    #           │   │
    # 1: ───H───X───@───M───
    #               │
    # 2: ───H───T───X───M───
    qreg = LineQubit.range(3)
    circ = Circuit(
        [ops.H.on_each(*qreg)],
        [ops.CNOT.on(qreg[0], qreg[1])],
        [ops.T.on(qreg[2])],
        [ops.TOFFOLI.on(*qreg)],
        [ops.measure_each(*qreg)],
    )
    folded = fold_global(circ, scale_factor=1.0)
    assert _equal(folded, circ)
    assert folded is not circ


def test_global_fold_stretch_factor_of_three():
    """Tests global folding with the scale as a factor of 3."""
    # Test circuit
    # 0: ───H───@───@───
    #           │   │
    # 1: ───H───X───@───
    #               │
    # 2: ───H───T───X───
    qreg = LineQubit.range(3)
    circ = Circuit(
        [ops.H.on_each(*qreg)],
        [ops.CNOT.on(qreg[0], qreg[1])],
        [ops.T.on(qreg[2])],
        [ops.TOFFOLI.on(*qreg)],
    )
    folded = fold_global(circ, scale_factor=3.0)
    correct = Circuit(circ, inverse(circ), circ)
    assert _equal(folded, correct)


def test_global_fold_stretch_factor_of_three_with_terminal_measurements():
    """Tests global folding with the scale as a factor of 3 for a circuit
    with terminal measurements.
    """
    # Test circuit
    # 0: ───H───@───@───M───
    #           │   │
    # 1: ───H───X───@───M───
    #               │
    # 2: ───H───T───X───M───
    qreg = LineQubit.range(3)
    circ = Circuit(
        [ops.H.on_each(*qreg)],
        [ops.CNOT.on(qreg[0], qreg[1])],
        [ops.T.on(qreg[2])],
        [ops.TOFFOLI.on(*qreg)],
    )
    meas = Circuit([ops.measure_each(*qreg)])
    folded = fold_global(circ + meas, scale_factor=3.0)
    correct = Circuit(circ, inverse(circ), circ, meas)
    assert _equal(folded, correct)


def test_global_fold_stretch_factor_nine_with_terminal_measurements():
    """Tests global folding with the scale as a factor of 9 for a circuit
    with terminal measurements.
    """
    # Test circuit
    # 0: ───H───@───@───M───
    #           │   │
    # 1: ───H───X───@───M───
    #               │
    # 2: ───H───T───X───M───
    qreg = LineQubit.range(3)
    circ = Circuit(
        [ops.H.on_each(*qreg)],
        [ops.CNOT.on(qreg[0], qreg[1])],
        [ops.T.on(qreg[2])],
        [ops.TOFFOLI.on(*qreg)],
    )
    meas = Circuit([ops.measure_each(*qreg)])
    folded = fold_global(circ + meas, scale_factor=9.0)
    correct = Circuit([circ, inverse(circ)] * 4, [circ], [meas])
    assert _equal(folded, correct)


def test_global_fold_stretch_factor_eight_terminal_measurements():
    """Tests global folding with a scale factor not a multiple of three so
    that local folding is also called.
    """
    # Test circuit
    # 0: ───H───@───@───M───
    #           │   │
    # 1: ───H───X───@───M───
    #               │
    # 2: ───H───T───X───M───
    qreg = LineQubit.range(3)
    circ = Circuit(
        [ops.H.on_each(*qreg)],
        [ops.CNOT.on(qreg[0], qreg[1])],
        [ops.T.on(qreg[2])],
        [ops.TOFFOLI.on(*qreg)],
    )
    meas = Circuit(ops.measure_each(*qreg))
    folded = fold_global(circ + meas, scale_factor=3.5)
    correct = Circuit(
        circ,
        inverse(circ),
        circ,
        inverse(Circuit([ops.T.on(qreg[2])], [ops.TOFFOLI.on(*qreg)],)),
        [ops.T.on(qreg[2])],
        [ops.TOFFOLI.on(*qreg)],
        meas,
    )
    assert _equal(folded, correct)


def test_convert_to_from_mitiq_qiskit():
    """Basic test for converting a Qiskit circuit to a Cirq circuit."""
    # Test Qiskit circuit:
    #          ┌───┐
    # q0_0: |0>┤ H ├──■──
    #          └───┘┌─┴─┐
    # q0_1: |0>─────┤ X ├
    #               └───┘
    qiskit_qreg = QuantumRegister(2)
    qiskit_circuit = QuantumCircuit(qiskit_qreg)
    qiskit_circuit.h(qiskit_qreg[0])
    qiskit_circuit.cnot(*qiskit_qreg)

    # Convert to a mitiq circuit
    mitiq_circuit, input_circuit_type = convert_to_mitiq(qiskit_circuit)
    assert isinstance(mitiq_circuit, Circuit)

    # Check correctness
    mitiq_qreg = LineQubit.range(2)
    correct_mitiq_circuit = Circuit(
        ops.H.on(mitiq_qreg[0]), ops.CNOT.on(*mitiq_qreg)
    )
    assert _equal(
        mitiq_circuit, correct_mitiq_circuit, require_qubit_equality=False
    )

    # Convert back to original circuit type
    original_circuit = convert_from_mitiq(mitiq_circuit, input_circuit_type)
    assert isinstance(original_circuit, QuantumCircuit)


def test_fold_from_left_with_qiskit_circuits():
    """Tests folding from left with Qiskit circuits."""
    # Test Qiskit circuit:
    #          ┌───┐
    # q0_0: |0>┤ H ├──■────■──
    #          ├───┤┌─┴─┐  │
    # q0_1: |0>┤ H ├┤ X ├──■──
    #          ├───┤├───┤┌─┴─┐
    # q0_2: |0>┤ H ├┤ T ├┤ X ├
    #          └───┘└───┘└───┘
    qiskit_qreg = QuantumRegister(3)
    qiskit_creg = ClassicalRegister(3)
    qiskit_circuit = QuantumCircuit(qiskit_qreg, qiskit_creg)
    qiskit_circuit.h(qiskit_qreg)
    qiskit_circuit.cnot(qiskit_qreg[0], qiskit_qreg[1])
    qiskit_circuit.t(qiskit_qreg[2])
    qiskit_circuit.ccx(*qiskit_qreg)
    qiskit_circuit.measure(qiskit_qreg, qiskit_creg)

    folded_circuit = fold_gates_from_left(
        qiskit_circuit, scale_factor=1.0, return_mitiq=True
    )

    qreg = LineQubit.range(3)
    correct_folded_circuit = Circuit(
        [ops.H.on_each(*qreg)],
        [ops.CNOT.on(qreg[0], qreg[1])],
        [ops.T.on(qreg[2])],
        [ops.TOFFOLI.on(*qreg)],
        [ops.measure_each(*qreg)],
    )

    assert isinstance(folded_circuit, Circuit)
    assert _equal(folded_circuit, correct_folded_circuit)

    # Keep the input type
    qiskit_folded_circuit = fold_gates_from_left(
        qiskit_circuit, scale_factor=1.0, return_mitiq=False
    )
    assert isinstance(qiskit_folded_circuit, QuantumCircuit)


def test_fold_from_right_with_qiskit_circuits():
    """Tests folding from right with Qiskit circuits."""
    # Test Qiskit circuit:
    #          ┌───┐
    # q0_0: |0>┤ H ├──■────■──
    #          ├───┤┌─┴─┐  │
    # q0_1: |0>┤ H ├┤ X ├──■──
    #          ├───┤├───┤┌─┴─┐
    # q0_2: |0>┤ H ├┤ T ├┤ X ├
    #          └───┘└───┘└───┘
    qiskit_qreg = QuantumRegister(3)
    qiskit_creg = ClassicalRegister(3)
    qiskit_circuit = QuantumCircuit(qiskit_qreg, qiskit_creg)
    qiskit_circuit.h(qiskit_qreg)
    qiskit_circuit.cnot(qiskit_qreg[0], qiskit_qreg[1])
    qiskit_circuit.t(qiskit_qreg[2])
    qiskit_circuit.ccx(*qiskit_qreg)
    qiskit_circuit.measure(qiskit_qreg, qiskit_creg)

    folded_circuit = fold_gates_from_right(
        qiskit_circuit, scale_factor=1.0, return_mitiq=True
    )

    qreg = LineQubit.range(3)
    correct_folded_circuit = Circuit(
        [ops.H.on_each(*qreg)],
        [ops.CNOT.on(qreg[0], qreg[1])],
        [ops.T.on(qreg[2])],
        [ops.TOFFOLI.on(*qreg)],
        [ops.measure_each(*qreg)],
    )

    assert isinstance(folded_circuit, Circuit)
    assert _equal(folded_circuit, correct_folded_circuit)

    # Keep the input type
    qiskit_folded_circuit = fold_gates_from_right(
        qiskit_circuit, scale_factor=1.0
    )
    assert isinstance(qiskit_folded_circuit, QuantumCircuit)


def test_fold_at_random_with_qiskit_circuits():
    """Tests folding at random with Qiskit circuits."""
    # Test Qiskit circuit:
    #          ┌───┐
    # q0_0: |0>┤ H ├──■────■──
    #          ├───┤┌─┴─┐  │
    # q0_1: |0>┤ H ├┤ X ├──■──
    #          ├───┤├───┤┌─┴─┐
    # q0_2: |0>┤ H ├┤ T ├┤ X ├
    #          └───┘└───┘└───┘
    qiskit_qreg = QuantumRegister(3)
    qiskit_creg = ClassicalRegister(3)
    qiskit_circuit = QuantumCircuit(qiskit_qreg, qiskit_creg)
    qiskit_circuit.h(qiskit_qreg)
    qiskit_circuit.cnot(qiskit_qreg[0], qiskit_qreg[1])
    qiskit_circuit.t(qiskit_qreg[2])
    qiskit_circuit.ccx(*qiskit_qreg)
    qiskit_circuit.measure(qiskit_qreg, qiskit_creg)

    folded_circuit = fold_gates_at_random(
        qiskit_circuit, scale_factor=1.0, return_mitiq=True
    )

    qreg = LineQubit.range(3)
    correct_folded_circuit = Circuit(
        [ops.H.on_each(*qreg)],
        [ops.CNOT.on(qreg[0], qreg[1])],
        [ops.T.on(qreg[2])],
        [ops.TOFFOLI.on(*qreg)],
        [ops.measure_each(*qreg)],
    )

    assert isinstance(folded_circuit, Circuit)
    assert _equal(folded_circuit, correct_folded_circuit)

    # Keep the input type
    qiskit_folded_circuit = fold_gates_at_random(
        qiskit_circuit, scale_factor=1.0
    )
    assert isinstance(qiskit_folded_circuit, QuantumCircuit)


def test_fold_global_with_qiskit_circuits():
    """Tests _fold_local with input Qiskit circuits."""
    # Test Qiskit circuit:
    #          ┌───┐
    # q0_0: |0>┤ H ├──■────■──
    #          ├───┤┌─┴─┐  │
    # q0_1: |0>┤ H ├┤ X ├──■──
    #          ├───┤├───┤┌─┴─┐
    # q0_2: |0>┤ H ├┤ T ├┤ X ├
    #          └───┘└───┘└───┘
    qiskit_qreg = QuantumRegister(3)
    qiskit_creg = ClassicalRegister(3)
    qiskit_circuit = QuantumCircuit(qiskit_qreg, qiskit_creg)
    qiskit_circuit.h(qiskit_qreg)
    qiskit_circuit.cnot(qiskit_qreg[0], qiskit_qreg[1])
    qiskit_circuit.t(qiskit_qreg[2])
    qiskit_circuit.ccx(*qiskit_qreg)
    qiskit_circuit.measure(qiskit_qreg, qiskit_creg)

    # Return mitiq circuit
    folded_circuit = fold_global(
        qiskit_circuit,
        scale_factor=2.71828,
        fold_method=fold_gates_from_left,
        return_mitiq=True,
    )
    assert isinstance(folded_circuit, Circuit)

    # Return input circuit type
    folded_qiskit_circuit = fold_global(
        qiskit_circuit, scale_factor=2.0, fold_method=fold_gates_from_left
    )
    assert isinstance(folded_qiskit_circuit, QuantumCircuit)


def test_fold_left_squash_moments():
    """Tests folding from left with kwarg squash_moments."""
    # Test circuit
    # 0: ───H───@───@───M───
    #           │   │
    # 1: ───H───X───@───M───
    #               │
    # 2: ───H───T───X───M───
    qreg = LineQubit.range(3)
    circ = Circuit(
        [ops.H.on_each(*qreg)],
        [ops.CNOT.on(qreg[0], qreg[1])],
        [ops.T.on(qreg[2])],
        [ops.TOFFOLI.on(*qreg)],
        [ops.measure_each(*qreg)],
    )
    folded_not_squashed = fold_gates_from_left(
        circ, scale_factor=3, squash_moments=False
    )
    folded_and_squashed = fold_gates_from_left(
        circ, scale_factor=3, squash_moments=True
    )
    correct = Circuit(
        [ops.H.on_each(*qreg)] * 3,
        [ops.CNOT.on(qreg[0], qreg[1])] * 3,
        [ops.T.on(qreg[2]), ops.T.on(qreg[2]) ** -1, ops.T.on(qreg[2])],
        [ops.TOFFOLI.on(*qreg)] * 3,
        [ops.measure_each(*qreg)],
    )
    assert _equal(folded_and_squashed, folded_not_squashed)
    assert _equal(folded_and_squashed, correct)
    assert len(folded_and_squashed) == 10


@pytest.mark.parametrize(
    "fold_method",
    [
        fold_gates_from_left,
        fold_gates_from_right,
        fold_gates_at_random,
        fold_global,
    ],
)
def test_fold_and_squash_max_stretch(fold_method):
    """Tests folding and squashing a two-qubit circuit."""
    # Test circuit:
    # 0: ───────H───────H───────H───────H───────H───
    #
    # 1: ───H───────H───────H───────H───────H───────

    # Get the test circuit
    d = 10
    qreg = LineQubit.range(2)
    circuit = Circuit()
    for i in range(d):
        circuit.insert(0, ops.H.on(qreg[i % 2]), strategy=InsertStrategy.NEW)
    folded_not_squashed = fold_method(
        circuit, scale_factor=3.0, squash_moments=False
    )
    folded_and_squashed = fold_method(
        circuit, scale_factor=3.0, squash_moments=True
    )
    folded_with_squash_moments_not_specified = fold_method(
        circuit, scale_factor=3.0
    )  # Checks that the default is to squash moments
    assert len(folded_not_squashed) == 30
    assert len(folded_and_squashed) == 15
    assert len(folded_with_squash_moments_not_specified) == 15


@pytest.mark.parametrize(
    "fold_method",
    [
        fold_gates_from_left,
        fold_gates_from_right,
        fold_gates_at_random,
        fold_global,
    ],
)
def test_fold_and_squash_random_circuits_random_stretches(fold_method):
    """Tests folding and squashing random circuits and ensures the number of
    moments in the squashed circuits is never greater than the number of
    moments in the un-squashed circuit.
    """
    rng = np.random.RandomState(seed=1)
    for _ in range(100):
        circuit = testing.random_circuit(
            qubits=8, n_moments=8, op_density=0.75
        )
        scale = 2 * rng.random() + 1
        folded_not_squashed = fold_method(
            circuit, scale_factor=scale, squash_moments=False
        )
        folded_and_squashed = fold_method(
            circuit, scale_factor=scale, squash_moments=True
        )
        assert len(folded_and_squashed) <= len(folded_not_squashed)


def test_default_weight():
    """Tests default weight of an n-qubit gates is 0.99**n."""
    qreg = LineQubit.range(3)
    assert np.isclose(_default_weight(ops.H.on(qreg[0])), 0.99)
    assert np.isclose(_default_weight(ops.CZ.on(qreg[0], qreg[1])), 0.9801)
    assert np.isclose(_default_weight(ops.TOFFOLI.on(*qreg[:3])), 0.970299)


def test_compute_weight_of_circuit():
    qreg = LineQubit.range(3)
    circ = Circuit(
        ops.H.on_each(*qreg),
        ops.CNOT.on(qreg[0], qreg[1]),
        ops.T.on(qreg[2]),
        ops.TOFFOLI.on(*qreg),
    )
    weight = _compute_weight(
        circ, weights={"H": 0.01, "CNOT": 0.05, "T": 0.0, "TOFFOLI": 0.1}
    )
    assert np.isclose(weight, 0.18)

    weight = _compute_weight(
        circ, weights={"single": 0.0, "double": 0.0, "triple": 0.1}
    )
    assert np.isclose(weight, 0.1)

    weight = _compute_weight(
        circ, weights={"single": 0.0, "H": 0.01, "CNOT": 0.1, "TOFFOLI": 0.2}
    )
    assert np.isclose(weight, 0.33)


@pytest.mark.parametrize(
    "fold_method",
    [fold_gates_from_left, fold_gates_from_right, fold_gates_at_random],
)
@pytest.mark.parametrize("qiskit", [True, False])
def test_fold_local_with_fidelities(fold_method, qiskit):
    qreg = LineQubit.range(3)
    circ = Circuit(
        ops.H.on_each(*qreg),
        ops.CNOT.on(qreg[0], qreg[1]),
        ops.T.on(qreg[2]),
        ops.TOFFOLI.on(*qreg),
    )
    if qiskit:
        circ = convert_from_mitiq(circ, "qiskit")
    # Only fold the Toffoli gate
    fidelities = {"H": 1.0, "T": 1.0, "CNOT": 1.0, "TOFFOLI": 0.95}
    folded = fold_method(circ, scale_factor=3.0, fidelities=fidelities)
    correct = Circuit(
        [ops.H.on_each(*qreg)],
        [ops.CNOT.on(qreg[0], qreg[1])],
        [ops.T.on(qreg[2])],
        [ops.TOFFOLI.on(*qreg)] * 3,
    )
    if qiskit:
        folded, _ = convert_to_mitiq(folded)
        assert equal_up_to_global_phase(folded.unitary(), correct.unitary())
    else:
        assert _equal(folded, correct)


@pytest.mark.parametrize(
    "fold_method",
    [fold_gates_from_left, fold_gates_from_right, fold_gates_at_random],
)
@pytest.mark.parametrize("qiskit", [True, False])
def test_fold_local_with_single_qubit_gates_fidelity_one(fold_method, qiskit):
    """Tests folding only two-qubit gates by using
    fidelities = {"single": 1.}.
    """
    qreg = LineQubit.range(3)
    circ = Circuit(
        ops.H.on_each(*qreg),
        ops.CNOT.on(qreg[0], qreg[1]),
        ops.T.on(qreg[2]),
        ops.TOFFOLI.on(*qreg),
    )
    if qiskit:
        circ = convert_from_mitiq(circ, "qiskit")
    folded = fold_method(
        circ,
        scale_factor=3.0,
        fidelities={"single": 1.0, "CNOT": 0.99, "TOFFOLI": 0.95},
    )
    correct = Circuit(
        [ops.H.on_each(*qreg)],
        [ops.CNOT.on(qreg[0], qreg[1])] * 3,
        [ops.T.on(qreg[2])],
        [ops.TOFFOLI.on(*qreg)] * 3,
    )
    if qiskit:
        folded, _ = convert_to_mitiq(folded)
        assert equal_up_to_global_phase(folded.unitary(), correct.unitary())
    else:
        assert _equal(folded, correct)


@pytest.mark.parametrize(
    "fold_method",
    [fold_gates_from_left, fold_gates_from_right, fold_gates_at_random],
)
@pytest.mark.parametrize("qiskit", [True, False])
def test_all_gates_folded_at_max_scale_with_fidelities(fold_method, qiskit):
    """Tests that all gates are folded regardless of the input fidelities when
    the scale factor is three.
    """
    rng = np.random.RandomState(1)

    qreg = LineQubit.range(3)
    circ = Circuit(
        ops.H.on_each(*qreg),
        ops.CNOT.on(qreg[0], qreg[1]),
        ops.T.on(qreg[2]),
        ops.TOFFOLI.on(*qreg),
    )
    ngates = len(list(circ.all_operations()))

    if qiskit:
        circ = convert_from_mitiq(circ, "qiskit")

    for _ in range(10):
        folded = fold_method(
            circ,
            scale_factor=3.0,
            fidelities={
                "H": rng.rand(),
                "T": rng.rand(),
                "CNOT": rng.rand(),
                "TOFFOLI": rng.rand(),
            },
        )
        correct = Circuit(
            [ops.H.on_each(*qreg)] * 3,
            [ops.CNOT.on(qreg[0], qreg[1])] * 3,
            [ops.T.on(qreg[2]), ops.T.on(qreg[2]) ** -1, ops.T.on(qreg[2])],
            [ops.TOFFOLI.on(*qreg)] * 3,
        )
        if qiskit:
            folded, _ = convert_to_mitiq(folded)
            assert equal_up_to_global_phase(
                folded.unitary(), correct.unitary()
            )
        else:
            assert _equal(folded, correct)
            assert len(list(folded.all_operations())) == 3 * ngates


@pytest.mark.parametrize(
    "fold_method",
    [fold_gates_from_left, fold_gates_from_right, fold_gates_at_random],
)
def test_fold_local_raises_error_with_bad_fidelities(fold_method):
    with pytest.raises(ValueError, match="Fidelities should be"):
        fold_method(Circuit(), scale_factor=1.21, fidelities={"H": -1.0})

    with pytest.raises(ValueError, match="Fidelities should be"):
        fold_method(Circuit(), scale_factor=1.21, fidelities={"CNOT": 0.0})

    with pytest.raises(ValueError, match="Fidelities should be"):
        fold_method(Circuit(), scale_factor=1.21, fidelities={"triple": 1.2})


@pytest.mark.parametrize("conversion_type", ("qiskit", "pyquil"))
def test_convert_from_mitiq_circuit_conversion_error(conversion_type):
    circuit = testing.random_circuit(qubits=5, n_moments=5, op_density=0.99)
    noisy = circuit.with_noise(ops.depolarize(p=0.1))

    with pytest.raises(
        CircuitConversionError, match="Circuit could not be converted from"
    ):
        convert_from_mitiq(noisy, conversion_type)


def test_convert_qiskit_to_mitiq_circuit_conversion_error():
    # Custom gates are not supported in conversions
    gate = Operator([[0.0, 1.0], [-1.0, 0.0]])
    qreg = QuantumRegister(1)
    circ = QuantumCircuit(qreg)
    circ.unitary(gate, [0])

    with pytest.raises(
        CircuitConversionError, match="Circuit could not be converted to"
    ):
        convert_to_mitiq(circ)


def test_convert_pyquil_to_mitiq_circuit_conversion_error():
    # Pragmas are not supported in conversions
    prog = Program(Pragma("INITIAL_REWIRING", ['"Partial"']))

    with pytest.raises(
        CircuitConversionError, match="Circuit could not be converted to"
    ):
        convert_to_mitiq(prog)


@pytest.mark.parametrize(
    "fold_method",
    (
        fold_gates_from_left,
        fold_gates_from_right,
        fold_gates_at_random,
        fold_global,
    ),
)
def test_folding_circuit_conversion_error_qiskit(fold_method):
    # Custom gates are not supported in conversions
    gate = Operator([[0.0, 1.0], [-1.0, 0.0]])
    qreg = QuantumRegister(1)
    circ = QuantumCircuit(qreg)
    circ.unitary(gate, [0])

    with pytest.raises(
        CircuitConversionError, match="Circuit could not be converted to"
    ):
        fold_method(circ, scale_factor=2.0)


@pytest.mark.parametrize(
    "fold_method",
    (
        fold_gates_from_left,
        fold_gates_from_right,
        fold_gates_at_random,
        fold_global,
    ),
)
def test_folding_circuit_conversion_error_pyquil(fold_method):
    # Pragmas are not supported in conversions
    prog = Program(Pragma("INITIAL_REWIRING", ['"Partial"']))

    with pytest.raises(
        CircuitConversionError, match="Circuit could not be converted to"
    ):
        fold_method(prog, scale_factor=2.0)
