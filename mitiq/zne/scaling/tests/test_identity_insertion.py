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

"""Unit tests for scaling noise by identity insertion."""

import numpy as np
import pytest
from cirq import (
    Circuit,
    LineQubit,
    ops,
    testing,
    equal_up_to_global_phase,
    GateOperation,
    InsertStrategy,
)
from cirq.google import Sycamore
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.quantum_info.operators import Operator
from pyquil import Program
from pyquil.quilbase import Pragma
from mitiq.conversions import (
    CircuitConversionError,
    convert_to_mitiq,
    convert_from_mitiq,
)
from cirq.ops import IdentityGate
from mitiq.utils import _equal
from mitiq.zne.scaling.folding import _create_weight_mask
from mitiq.zne.scaling.identity_insertion import (
    UnscalableCircuitError,
    _apply_scale_mask,
    _create_scale_mask,
    scale_gates_from_left,
    scale_gates_from_right,
    scale_gates_at_random,
)


@pytest.mark.parametrize(
    "scale_method",
    [scale_gates_at_random, scale_gates_from_left, scale_gates_from_right,],
)
def test_scaling_with_bad_scale_factor(scale_method):
    """Checks the test fails when scale factor is negative and input method is
     called directly."""
    with pytest.raises(ValueError, match="Requires scale_factor >= 1.0"):
        scale_method(Circuit(), scale_factor=-1.0)


@pytest.mark.parametrize("method", ["at_random", "from_right", "from_left"])
def test_create_mask_with_bad_scale_factor(method):
    """Checks the test fails when scale factor is negative and input method is
    a string."""
    with pytest.raises(ValueError, match="Requires scale_factor >= 1.0"):
        _create_scale_mask([1], scale_factor=-1.0, scaling_method=method)


def test_create_mask_with_bad_scaling_method():
    "Checks test fails when no scaling method is given."
    with pytest.raises(ValueError, match="'scaling_method' is not valid."):
        _create_scale_mask([1], scale_factor=1.5, scaling_method=None)

@pytest.mark.parametrize("method", ("at_random", "from_left", "from_right"))
def test_create_scale_mask_approximates_well(method):
    """Check _create_scale_mask well approximates the scale factor."""
    rnd_state = np.random.RandomState(seed=0)
    for scale_factor in [1.1, 1.4, 1.8, 2.8, 6.8, 18.8, 19.0, 31]:
        weight_mask = [rnd_state.rand() for _ in range(100)]
        seed = rnd_state.randint(100)
        fold_mask = _create_scale_mask(
            weight_mask, scale_factor, scaling_method=method, seed=seed,
        )
        out_weights = [w + n * w for w, n in zip(weight_mask, fold_mask)]
        actual_scale = sum(out_weights) / sum(weight_mask)
        # Less than 10% error
        assert np.isclose(scale_factor/actual_scale, 1.0, atol=0.1)


def test_create_scale_mask_with_fidelities():
    qreg = LineQubit.range(3)
    circ = Circuit(
        ops.H.on_each(*qreg),
        ops.CNOT.on(qreg[0], qreg[1]),
        ops.T.on(qreg[2]),
        ops.TOFFOLI.on(*qreg),
        ops.measure_each(*qreg),
    )
    # Measurement gates should be ignored
    fidelities = {"H": 0.9, "CNOT": 0.8, "T": 0.7, "TOFFOLI": 0.6}
    weight_mask = _create_weight_mask(circ, kwargs.get("fidelities"))
    scale_mask = _create_scale_mask(circ, weight_mask)
    assert np.allclose(scale_mask, [0.1, 0.1, 0.1, 0.2, 0.3, 0.4])

    fidelities = {"single": 1.0, "double": 0.5, "triple": 0.1}
    weight_mask = _create_weight_mask(circ, kwargs.get("fidelities"))
    scale_mask = _create_scale_mask(circ, weight_mask)
    assert np.allclose(scale_mask, [0.0, 0.0, 0.0, 0.5, 0.0, 0.9])

    fidelities = {"single": 1.0, "H": 0.1, "CNOT": 0.2, "TOFFOLI": 0.3}
    weight_mask = _create_weight_mask(circ, kwargs.get("fidelities"))
    scale_mask = _create_scale_mask(circ, weight_mask)
    assert np.allclose(scale_mask, [0.9, 0.9, 0.9, 0.8, 0.0, 0.7])

@pytest.mark.parametrize(
    "scale_method",
    [scale_gates_from_left, scale_gates_from_right, scale_gates_at_random],
)
def test_scale_no_stretch(scale_method):
    """Unit test for scaling gates from left for a scale factor of 0 i.e
    no identity gates are inserted."""
    circuit = testing.random_circuit(qubits=2, n_moments=10, op_density=0.99)
    scaled = scale_method(circuit, scale_factor=1.0, seed=None)
    assert _equal(scaled, circuit)
    assert not (scaled is circuit)



def test_scale_from_left_very_small_factor():
    """Basic test for scaling from left with very small scale factor.
    """
    # Test Circuit
    # 0: ───H───@───────
    #           |
    # 1: ───H───X───T───
    qreg = LineQubit.range(2)
    circ = Circuit(
        [
            ops.H.on_each(*qreg),
            ops.CNOT.on(qreg[0], qreg[1]),
            ops.T.on(qreg[1]),
        ]
    )

    scaled = scale_gates_from_left(circ, scale_factor=1.1, seed=3)
    correct = Circuit(
        [
            ops.H.on_each(*qreg),
            ops.CNOT.on(qreg[0], qreg[1]),
            ops.T.on(qreg[1]),
        ]
    )
    assert _equal(scaled, correct)



def test_scale_from_left_small_factor():
    """Basic test for scaling from left with small scale factor.
    """
    # Test Circuit
    # 0: ───H───@───────
    #           |
    # 1: ───H───X───T───
    qreg = LineQubit.range(2)
    circ = Circuit(
        [
            ops.H.on_each(*qreg),
            ops.CNOT.on(qreg[0], qreg[1]),
            ops.T.on(qreg[1]),
        ]
    )

    scaled = scale_gates_from_left(circ, scale_factor=1.5, seed=3)
    correct = Circuit(
        [
            ops.H.on_each(*qreg),
            ops.I.on_each(*qreg),
            ops.CNOT.on(qreg[0], qreg[1]),
            ops.T.on(qreg[1]),
        ]
    )
    assert _equal(scaled, correct)

def test_scale_from_left_intermediate_factor():
    """Basic test for scaling from left with intermediate scale factor.
    """
    # Test Circuit
    # 0: ───H───@───────
    #           |
    # 1: ───H───X───T───
    qreg = LineQubit.range(2)
    circ = Circuit(
        [
            ops.H.on_each(*qreg),
            ops.CNOT.on(qreg[0], qreg[1]),
            ops.T.on(qreg[1]),
        ]
    )

    scaled = scale_gates_from_left(circ, scale_factor=1.7, seed=3)
    correct = Circuit(
        [
            ops.H.on_each(*qreg),
            ops.I.on_each(*qreg),
            ops.CNOT.on(qreg[0], qreg[1]),
            ops.T.on(qreg[1]),
        ]
    )
    assert _equal(scaled, correct)

def test_scale_from_right_very_small_factor():
    """Basic test for scaling from right with very small scale factor.
    """
    # Test Circuit
    # 0: ───H───@───────
    #           |
    # 1: ───H───X───T───
    qreg = LineQubit.range(2)
    circ = Circuit(
        [
            ops.H.on_each(*qreg),
            ops.CNOT.on(qreg[0], qreg[1]),
            ops.T.on(qreg[1]),
        ]
    )

    scaled = scale_gates_from_right(circ, scale_factor=1.1, seed=3)
    correct = Circuit(
        [
            ops.H.on_each(*qreg),
            ops.CNOT.on(qreg[0], qreg[1]),
            ops.T.on(qreg[1]),
        ]
    )
    assert _equal(scaled, correct)



def test_scale_from_right_small_factor():
    """Basic test for scaling from right with small scale factor.
    """
    # Test Circuit
    # 0: ───H───@───────
    #           |
    # 1: ───H───X───T───
    qreg = LineQubit.range(2)
    circ = Circuit(
        [
            ops.H.on_each(*qreg),
            ops.CNOT.on(qreg[0], qreg[1]),
            ops.T.on(qreg[1]),
        ]
    )

    scaled = scale_gates_from_right(circ, scale_factor=1.5, seed=3)
    correct = Circuit(
        [
            ops.H.on_each(*qreg),
            ops.I.on_each(*qreg),
            ops.CNOT.on(qreg[0], qreg[1]),
            ops.T.on(qreg[1]),
        ]
    )
    assert _equal(scaled, correct)

def test_scale_from_right_intermediate_factor():
    """Basic test for scaling from right with intermediate scale factor.
    """
    # Test Circuit
    # 0: ───H───@───────
    #           |
    # 1: ───H───X───T───
    qreg = LineQubit.range(2)
    circ = Circuit(
        [
            ops.H.on_each(*qreg),
            ops.CNOT.on(qreg[0], qreg[1]),
            ops.T.on(qreg[1]),
        ]
    )

    scaled = scale_gates_from_right(circ, scale_factor=1.7, seed=3)
    correct = Circuit(
        [
            ops.H.on_each(*qreg),
            ops.I.on_each(*qreg),
            ops.CNOT.on(qreg[0], qreg[1]),
            ops.T.on(qreg[1]),
        ]
    )
    assert _equal(scaled, correct)

def test_scale_at_random_with_very_small_factor():
    """Basic test for scaling at random with very small scale factor.
    """
    # Test Circuit
    # 0: ───H───@───────
    #           |
    # 1: ───H───X───T───
    qreg = LineQubit.range(2)
    circ = Circuit(
        [
            ops.H.on_each(*qreg),
            ops.CNOT.on(qreg[0], qreg[1]),
            ops.T.on(qreg[1]),
        ]
    )

    scaled = scale_gates_at_random(circ, scale_factor=1.1, seed=3)
    correct = Circuit(
        [
            ops.H.on_each(*qreg),
            ops.CNOT.on(qreg[0], qreg[1]),
            ops.T.on(qreg[1]),
        ]
    )
    assert _equal(scaled, correct)



def test_scale_scale_gates_at_random_with_small_factor():
    """Basic test for scaling at random with small scale factor.
    """
    # Test Circuit
    # 0: ───H───@───────
    #           |
    # 1: ───H───X───T───
    qreg = LineQubit.range(2)
    circ = Circuit(
        [
            ops.H.on_each(*qreg),
            ops.CNOT.on(qreg[0], qreg[1]),
            ops.T.on(qreg[1]),
        ]
    )

    scaled = scale_gates_at_random(circ, scale_factor=1.5, seed=3)
    correct = Circuit(
        [
            ops.H.on_each(*qreg),
            ops.I.on_each(*qreg),
            ops.CNOT.on(qreg[0], qreg[1]),
            ops.T.on(qreg[1]),
        ]
    )
    assert _equal(scaled, correct)

def test_scale_scale_gates_at_random_with_intermediate_factor():
    """Basic test for scaling at random with intermediate scale factor.
    """
    # Test Circuit
    # 0: ───H───@───────
    #           |
    # 1: ───H───X───T───
    qreg = LineQubit.range(2)
    circ = Circuit(
        [
            ops.H.on_each(*qreg),
            ops.CNOT.on(qreg[0], qreg[1]),
            ops.T.on(qreg[1]),
        ]
    )

    scaled = scale_gates_at_random(circ, scale_factor=1.7, seed=3)
    correct = Circuit(
        [
            ops.H.on_each(*qreg),
            ops.I.on_each(*qreg),
            ops.CNOT.on(qreg[0], qreg[1]),
            ops.T.on(qreg[1]),
        ]
    )
    assert _equal(scaled, correct)


@pytest.mark.parametrize(
    "scale_method",
    [scale_gates_from_left, scale_gates_from_right, scale_gates_at_random],
)
def test_scale_with_full_factor(scale_method):
    """Basic test for scaling from left and right with full
    scale factor.
    """
    # Test Circuit
    # 0: ───H───@───────
    #           |
    # 1: ───H───X───T───
    qreg = LineQubit.range(2)
    circ = Circuit(
        [
            ops.H.on_each(*qreg),
            ops.CNOT.on(qreg[0], qreg[1]),
            ops.T.on(qreg[1]),
        ]
    )

    scaled = scale_method(circ, scale_factor=2.0, seed=3)
    correct = Circuit(
        [
            ops.H.on_each(*qreg),
            [ops.I.on_each(*qreg)] * 2,
            ops.CNOT.on(qreg[0], qreg[1]),
            [GateOperation(IdentityGate(2), qreg)] * 2,
            ops.T.on(qreg[1]),
            [ops.I.on(qreg[1])] * 2,
        ]
    )
    assert _equal(scaled, correct)


@pytest.mark.parametrize(
    "scale_method",
    [scale_gates_from_left, scale_gates_from_right, scale_gates_at_random],
)
def test_scale_from_left_and_right_three_qubits(scale_method):
    """Unit test for scaling gates from left to for a 3 qubit circuit."""
    # Test Circuit
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

    scaled = scale_method(circ, scale_factor=2, seed=3)
    correct = Circuit(
        [
            ops.H.on_each(*qreg),
            [ops.I.on_each(*qreg)] * 2,
            ops.CNOT.on(qreg[0], qreg[1]),
            [GateOperation(IdentityGate(2), (qreg[0], qreg[1]))] * 2,
            ops.T.on(qreg[2]),
            [ops.I.on(qreg[2])] * 2,
            ops.TOFFOLI.on(*qreg),
            [GateOperation(IdentityGate(3), qreg)] * 2,
        ]
    )
    assert _equal(scaled, correct)


@pytest.mark.parametrize(
    "scale_method",
    [scale_gates_from_left, scale_gates_from_right, scale_gates_at_random],
)
def test_scale_from_left_and_right_scale_factor_more_than_full_factor(
    scale_method,
):
    """Tests scaling from left with a scale_factor larger than two."""
    qreg = LineQubit.range(2)
    circuit = Circuit([ops.SWAP.on(*qreg)], [ops.CNOT.on(*qreg)])
    scaled = scale_method(circuit, scale_factor=5.0)
    correct = Circuit(
        [ops.SWAP.on(*qreg)],
        [GateOperation(IdentityGate(2), qreg)] * 5,
        [ops.CNOT.on(*qreg)],
        [GateOperation(IdentityGate(2), qreg)] * 5,
    )
    assert _equal(scaled, correct)


@pytest.mark.parametrize(
    "scale_method",
    [scale_gates_from_left, scale_gates_from_right, scale_gates_at_random],
)
def test_scale_from_left_and_right_with_terminal_measurements_no_stretch(
    scale_method,
):
    """Tests scaling from left with terminal measurements."""
    qreg = LineQubit.range(3)
    circ = Circuit(
        [ops.H.on_each(*qreg)],
        [ops.CNOT.on(qreg[0], qreg[1])],
        [ops.T.on(qreg[2])],
        [ops.TOFFOLI.on(*qreg)],
        [ops.measure_each(*qreg)],
    )
    scaled = scale_method(circ, scale_factor=1.0)
    correct = Circuit(
        [ops.H.on_each(*qreg)],
        [ops.CNOT.on(qreg[0], qreg[1])],
        [ops.T.on(qreg[2])],
        [ops.TOFFOLI.on(*qreg)],
        [ops.measure_each(*qreg)],
    )
    assert _equal(scaled, correct)


@pytest.mark.parametrize(
    "scale_method",
    [scale_gates_from_left, scale_gates_from_right, scale_gates_at_random],
)
def test_fold_from_left_and_right_with_terminal_measurements_max_stretch(
    scale_method,
):
    """Tests scaling from left with terminal measurements."""
    qreg = LineQubit.range(3)
    circ = Circuit(
        [ops.H.on_each(*qreg)],
        [ops.CNOT.on(qreg[0], qreg[1])],
        [ops.T.on(qreg[2])],
        [ops.TOFFOLI.on(*qreg)],
        [ops.measure_each(*qreg)],
    )
    scaled = scale_method(circ, scale_factor=2.0)
    correct = Circuit(
        [ops.H.on_each(*qreg)],
        [ops.I.on_each(*qreg)] * 2,
        [ops.CNOT.on(qreg[0], qreg[1])],
        [GateOperation(IdentityGate(2), (qreg[0], qreg[1]))] * 2,
        [ops.T.on(qreg[2])],
        [ops.I.on(qreg[2])] * 2,
        [ops.TOFFOLI.on(*qreg)],
        [GateOperation(IdentityGate(3), (qreg))] * 2,
        [ops.measure_each(*qreg)],
    )
    assert _equal(scaled, correct)

    # Make sure original circuit is not modified
    original = Circuit(
        [ops.H.on_each(*qreg)],
        [ops.CNOT.on(qreg[0], qreg[1])],
        [ops.T.on(qreg[2])],
        [ops.TOFFOLI.on(*qreg)],
        [ops.measure_each(*qreg)],
    )
    assert _equal(circ, original)


@pytest.mark.parametrize(
    "scale_method",
    [scale_gates_from_left, scale_gates_from_right, scale_gates_at_random,],
)
def test_scale_with_intermediate_measurements_raises_error(scale_method):
    """Tests local scaling functions raise an error on circuits with
    intermediate measurements.
    """
    qbit = LineQubit(0)
    circ = Circuit([ops.H.on(qbit)], [ops.measure(qbit)], [ops.T.on(qbit)])
    with pytest.raises(
        UnscalableCircuitError,
        match="Circuit contains intermediate measurements",
    ):
        scale_method(circ, scale_factor=3.0)


@pytest.mark.parametrize(
    "scale_method",
    [scale_gates_from_left, scale_gates_from_right, scale_gates_at_random],
)
def test_scale_with_channels_raises_error(scale_method):
    """Tests local scaling functions raise an error on circuits with
    non-unitary channels (which are not measurements).
    """
    qbit = LineQubit(0)
    circ = Circuit(
        ops.H.on(qbit), ops.depolarize(p=0.1).on(qbit), ops.measure(qbit)
    )
    with pytest.raises(
        UnscalableCircuitError, match="Circuit contains non-unitary channels"
    ):
        scale_method(circ, scale_factor=3.0)


@pytest.mark.parametrize(
    "scale_method",
    [scale_gates_from_left, scale_gates_from_right, scale_gates_at_random],
)
def test_scale_no_repeats(scale_method):
    """Tests scaling at random to ensure that no gates are folded twice and
    scaled gates are not scaled again.
    """
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
        scaled = scale_method(circ, scale_factor=scale, seed=1)
        gates = list(scaled.all_operations())
        counts = {gate: gates.count(gate) for gate in circuit_ops}
        assert all(count <= 3 for count in counts.values())


def test_scale_right_retains_terminal_measurements_in_input_circuit():
    """Tests that scaling from the right doesn't modify the terminal
    measurements in the input circuit.
    """
    qbit = LineQubit(1)
    circ = Circuit(ops.H.on(qbit), ops.measure(qbit))
    scaled = scale_gates_from_right(circ, scale_factor=1.0)
    assert _equal(circ, scaled)


def test_local_scaling_methods_match_on_even_scale_factors():
    circuit = testing.random_circuit(
        qubits=3, n_moments=5, op_density=1.0, random_state=11
    )
    for s in (2, 6, 14):
        assert _equal(
            scale_gates_from_left(circuit, s),
            scale_gates_from_right(circuit, s),
            require_qubit_equality=True,
            require_measurement_equality=True,
        )

    for s in (2, 6, 14):
        assert _equal(
            scale_gates_from_left(circuit, s),
            scale_gates_at_random(circuit, s),
            require_qubit_equality=True,
            require_measurement_equality=True,
        )

@pytest.mark.parametrize(
    "scale_method",
    [scale_gates_from_left, scale_gates_from_right, scale_gates_at_random],
)
def test_local_no_scale_with_qiskit_circuits(scale_method):
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

    folded_circuit = scale_method(
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
    qiskit_folded_circuit = scale_method(
        qiskit_circuit, scale_factor=1.0, return_mitiq=False
    )
    assert isinstance(qiskit_folded_circuit, QuantumCircuit)
    assert qiskit_folded_circuit.qregs == qiskit_circuit.qregs
    assert qiskit_folded_circuit.cregs == qiskit_circuit.cregs

@pytest.mark.parametrize(
    "scale_method",
    (
        scale_gates_from_left,
        scale_gates_from_right,
        scale_gates_at_random,
    ),
)
def test_folding_circuit_conversion_error_qiskit(scale_method):
    # Custom gates are not supported in conversions
    gate = Operator([[0.0, 1.0], [-1.0, 0.0]])
    qreg = QuantumRegister(1)
    circ = QuantumCircuit(qreg)
    circ.unitary(gate, [0])

    with pytest.raises(
        CircuitConversionError, match="Circuit could not be converted to"
    ):
        scale_method(circ, scale_factor=2.0)

@pytest.mark.parametrize(
    "scale_method",
    (
        scale_gates_from_left,
        scale_gates_from_right,
        scale_gates_at_random,
    ),
)
def test_folding_circuit_conversion_error_pyquil(scale_method):
    # Pragmas are not supported in conversions
    prog = Program(Pragma("INITIAL_REWIRING", ['"Partial"']))

    with pytest.raises(
        CircuitConversionError, match="Circuit could not be converted to"
    ):
        scale_method(prog, scale_factor=2.0)


@pytest.mark.parametrize(
    "scale_method",
    (
        scale_gates_from_left,
        scale_gates_from_right,
        scale_gates_at_random,
    ),
)
def test_scale_local_squash_moments(scale_method):
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
    folded_not_squashed = scale_method(
        circ, scale_factor=3, squash_moments=False
    )
    folded_and_squashed = scale_method(
        circ, scale_factor=3, squash_moments=True
    )
    correct = Circuit(
        [ops.H.on_each(*qreg)],
        [ops.I.on_each(*qreg)] * 3,
        [ops.CNOT.on(qreg[0], qreg[1])],
        [GateOperation(IdentityGate(2), (qreg[0], qreg[1]))] * 3,
        [ops.T.on(qreg[2])],
        [ops.I.on(qreg[2])] * 3,
        [ops.TOFFOLI.on(*qreg)],
        [GateOperation(IdentityGate(3), qreg)] * 3,
        [ops.measure_each(*qreg)],
    )
    assert _equal(folded_and_squashed, folded_not_squashed)
    assert _equal(folded_and_squashed, correct)
    assert len(folded_and_squashed) == 13

@pytest.mark.parametrize(
    "scale_method",
    [
        scale_gates_from_left,
        scale_gates_from_right,
        scale_gates_at_random,
    ],
)
def test_fold_and_squash_max_stretch(scale_method):
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

    folded_not_squashed = scale_method(
        circuit, scale_factor=2.0, squash_moments=False
    )
    folded_and_squashed = scale_method(
        circuit, scale_factor=2.0, squash_moments=True
    )
    folded_with_squash_moments_not_specified = scale_method(
        circuit, scale_factor=2.0
    )  # Checks that the default is to squash moments

    assert len(folded_not_squashed) == 30
    assert len(folded_and_squashed) == 15
    assert len(folded_with_squash_moments_not_specified) == 15


@pytest.mark.parametrize(
    "scale_method",
    [
        scale_gates_from_left,
        scale_gates_from_right,
        scale_gates_at_random,
    ],
)
def test_scale_and_squash_random_circuits_random_stretches(scale_method):
    """Tests folding and squashing random circuits and ensures the number of
    moments in the squashed circuits is never greater than the number of
    moments in the un-squashed circuit.
    """
    rng = np.random.RandomState(seed=1)
    for trial in range(5):
        circuit = testing.random_circuit(
            qubits=8, n_moments=8, op_density=0.75
        )
        scale = 2 * rng.random() + 1
        folded_not_squashed = scale_method(
            circuit, scale_factor=scale, squash_moments=False, seed=trial,
        )
        folded_and_squashed = scale_method(
            circuit, scale_factor=scale, squash_moments=True, seed=trial,
        )
        assert len(folded_and_squashed) <= len(folded_not_squashed)

@pytest.mark.parametrize(
    "scale_method",
    [scale_gates_from_left, scale_gates_from_right, scale_gates_at_random],
)
@pytest.mark.parametrize("qiskit", [True, False])
def test_scale_local_with_fidelities(scale_method, qiskit):
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
    folded = scale_method(circ, scale_factor=3.0, fidelities=fidelities)
    correct = Circuit(
        [ops.H.on_each(*qreg)],
        [ops.CNOT.on(qreg[0], qreg[1])],
        [ops.T.on(qreg[2])],
        [ops.TOFFOLI.on(*qreg)],
        [GateOperation(IdentityGate(3), qreg)] * 3,
    )
    if qiskit:
        folded, _ = convert_to_mitiq(folded)
        assert equal_up_to_global_phase(folded.unitary(), correct.unitary())
    else:
        assert _equal(folded, correct)

@pytest.mark.parametrize(
    "scale_method",
    [scale_gates_from_left, scale_gates_from_right, scale_gates_at_random],
)
@pytest.mark.parametrize("qiskit", [True, False])
def test_scale_local_with_single_qubit_gates_fidelity_one(scale_method, qiskit):
    """Tests folding only two-qubit gates by using fidelities = {"single": 1.}.
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
    folded = scale_method(
        circ,
        scale_factor=3.0,
        fidelities={"single": 1.0, "CNOT": 0.99, "TOFFOLI": 0.95},
    )
    correct = Circuit(
        [ops.H.on_each(*qreg)],
        [ops.CNOT.on(qreg[0], qreg[1])],
        [GateOperation(IdentityGate(2), (qreg[0], qreg[1]))] * 3,
        [ops.T.on(qreg[2])],
        [ops.TOFFOLI.on(*qreg)],
        [GateOperation(IdentityGate(3), qreg)] * 3,
    )
    if qiskit:
        assert folded.qregs == circ.qregs
        assert folded.cregs == circ.cregs
        folded, _ = convert_to_mitiq(folded)
        assert equal_up_to_global_phase(folded.unitary(), correct.unitary())
    else:
        assert _equal(folded, correct)

@pytest.mark.parametrize(
    "scale_method",
    [scale_gates_from_left, scale_gates_from_right, scale_gates_at_random],
)
@pytest.mark.parametrize("qiskit", [True, False])
def test_all_gates_folded_at_max_scale_with_fidelities(scale_method, qiskit):
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

    folded = scale_method(
        circ,
        scale_factor=2.0,
        fidelities={
            "H": rng.rand(),
            "T": rng.rand(),
            "CNOT": rng.rand(),
            "TOFFOLI": rng.rand(),
        },
    )
    correct = Circuit(
        [ops.H.on_each(*qreg)],
        [ops.I.on_each(*qreg)] * 2,
        [ops.CNOT.on(qreg[0], qreg[1])],
        [GateOperation(IdentityGate(2), (qreg[0], qreg[1]))] * 2,
        [ops.T.on(qreg[2])],
        [ops.I.on(qreg[2])] * 2,
        [ops.TOFFOLI.on(*qreg)],
        [GateOperation(IdentityGate(3), qreg)] * 2,
    )
    if qiskit:
        assert folded.qregs == circ.qregs
        assert folded.cregs == circ.cregs
        folded, _ = convert_to_mitiq(folded)
        assert equal_up_to_global_phase(folded.unitary(), correct.unitary())
    else:
        assert _equal(folded, correct)
        assert len(list(folded.all_operations())) == 3 * ngates

@pytest.mark.parametrize(
    "scale_method",
    [scale_gates_from_left, scale_gates_from_right, scale_gates_at_random],
)
def test_scale_local_raises_error_with_bad_fidelities(scale_method):
    with pytest.raises(ValueError, match="Fidelities should be"):
        scale_method(Circuit(), scale_factor=1.21, fidelities={"H": -1.0})

    with pytest.raises(ValueError, match="Fidelities should be"):
        scale_method(Circuit(), scale_factor=1.21, fidelities={"CNOT": 0.0})

    with pytest.raises(ValueError, match="Fidelities should be"):
        scale_method(Circuit(), scale_factor=1.21, fidelities={"triple": 1.2})

@pytest.mark.parametrize(
    "scale_method",
    [scale_gates_from_left, scale_gates_from_right, scale_gates_at_random],
)
@pytest.mark.parametrize("scale", [1, 3, 5, 9])
def test_scale_fidelity_large_scale_factor_only_twoq_gates(scale_method, scale):
    qreg = LineQubit.range(2)
    circuit = Circuit(ops.H(qreg[0]), ops.CNOT(*qreg))
    folded = scale_method(
        circuit, scale_factor=scale, fidelities={"single": 1.0}
    )
    correct = Circuit(ops.H(qreg[0]), [ops.CNOT(*qreg)], [GateOperation(IdentityGate(2), qreg)] * scale)
    assert _equal(folded, correct)


def test_scaling_keeps_measurement_order_with_qiskit():
    qreg, creg = QuantumRegister(2), ClassicalRegister(2)
    circuit = QuantumCircuit(qreg, creg)
    circuit.h(qreg[0])
    circuit.measure(qreg, creg)

    folded = scale_gates_at_random(circuit, scale_factor=1.0)
    assert folded == circuit

@pytest.mark.parametrize(
    "weight_mask",
    ([0.1, 0.2, 0.3, 0.0], [0.3, 0.5, 0.7, 0.0], [1.0, 1.0, 1.0, 0.0]),
)
@pytest.mark.parametrize("scale_factor", (1, 3, 5, 7, 9, 11))
@pytest.mark.parametrize("method", ("at_random", "from_left", "from_right"))
def test_create_fold_mask_with_odd_scale_factors(
    weight_mask, scale_factor, method,
):
    fold_mask = _create_scale_mask(weight_mask, scale_factor, method)
    num_folds = int((scale_factor))
    assert fold_mask == [num_folds, num_folds, num_folds, 0]

@pytest.mark.parametrize(
    "weight_mask",
    ([0.1, 0.2, 0.3, 0.0], [0.3, 0.5, 0.7, 0.0], [1.0, 1.0, 1.0, 0.0]),
)
@pytest.mark.parametrize("scale_factor", (2, 4, 6, 8, 10, 12))
@pytest.mark.parametrize("method", ("at_random", "from_left", "from_right"))
def test_create_fold_mask_with_even_scale_factors(
    weight_mask, scale_factor, method,
):
    fold_mask = _create_scale_mask(weight_mask, scale_factor, method)
    num_folds = int((scale_factor))
    assert fold_mask == [num_folds, num_folds, num_folds, 0]

@pytest.mark.parametrize("method", ("at_random", "from_left", "from_right"))
def test_create_scale_mask_with_real_scale_factors(method):
    fold_mask = _create_scale_mask(
        weight_mask=[0.1, 0.2, 0.3, 0.0],
        scale_factor=1.0,
        scaling_method=method,
    )
    assert fold_mask == [0, 0, 0, 0]

    fold_mask = _create_scale_mask(
        weight_mask=[0.1, 0.1, 0.1, 0.1],
        scale_factor=1.5,
        scaling_method=method,
    )
    assert fold_mask == [1, 1, 1, 1]

    fold_mask = _create_scale_mask(
        weight_mask=[1.0, 1.0, 1.0, 1.0],
        scale_factor=2,
        scaling_method=method,
    )
    assert fold_mask == [2, 2, 2, 2]

    fold_mask = _create_scale_mask(
        weight_mask=[1.0, 1.0, 1.0, 1.0],
        scale_factor=3.9,
        scaling_method=method,
    )
    assert fold_mask == [3, 3, 3, 3]

def test_apply_scale_mask_wrong_size():
    qreg = LineQubit(0)
    circ = Circuit(ops.H(qreg))
    with pytest.raises(ValueError, match="have incompatible sizes"):
        _ = _apply_scale_mask(circ, [1, 1])
