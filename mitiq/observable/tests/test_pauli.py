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

import pytest

import numpy as np
import cirq

from mitiq.observable import PauliString, Observable
from mitiq.interface import mitiq_qiskit, mitiq_pyquil
from mitiq.utils import _equal


# Basis rotations to measure Pauli X and Y.
xrotation = cirq.SingleQubitCliffordGate.Y_nsqrt
yrotation = cirq.SingleQubitCliffordGate.X_sqrt


def test_pauli_init():
    pauli = PauliString(spec="IZXYI", coeff=1.0)
    a, b, c = cirq.LineQubit.range(1, 4)
    assert pauli._pauli == cirq.PauliString(
        1.0, cirq.Z(a), cirq.X(b), cirq.Y(c)
    )
    assert str(pauli) == "Z(1)*X(2)*Y(3)"


def test_pauli_init_with_support():
    support = (2, 37)
    pauli = PauliString(spec="XZ", support=support)

    assert pauli._pauli == cirq.PauliString(
        cirq.X(cirq.LineQubit(support[0])), cirq.Z(cirq.LineQubit(support[1]))
    )
    assert str(pauli) == f"X({support[0]})*Z({support[1]})"


def test_matrix():
    xmat = cirq.unitary(cirq.X)
    zmat = cirq.unitary(cirq.Z)

    assert np.allclose(PauliString(spec="X").matrix(), xmat)
    assert np.allclose(PauliString(spec="Z", coeff=-0.5).matrix(), -0.5 * zmat)
    assert np.allclose(PauliString(spec="ZZ").matrix(), np.kron(zmat, zmat))
    assert np.allclose(PauliString(spec="XZ").matrix(), np.kron(xmat, zmat))


def test_pauli_matrix_include_qubits():
    imat = np.identity(2)
    xmat = cirq.unitary(cirq.X)
    pauli = PauliString(spec="X")

    assert np.allclose(pauli.matrix(), xmat)
    assert np.allclose(
        pauli.matrix(qubit_indices_to_include=[0, 1]), np.kron(xmat, imat)
    )
    assert np.allclose(
        pauli.matrix(qubit_indices_to_include=[0, 1, 2]),
        np.kron(np.kron(xmat, imat), imat),
    )


@pytest.mark.parametrize("support", [range(3), range(1, 4)])
@pytest.mark.parametrize("circuit_type", ("cirq", "qiskit", "pyquil"))
def test_pauli_measure_in_circuit(support, circuit_type):
    pauli = PauliString(spec="XYZ", support=support, coeff=-0.5)

    names = ("0th", "1st", "2nd", "3rd", "4th", "5th")
    qreg = [cirq.NamedQubit(name) for name in names]
    base_circuit = cirq.Circuit(cirq.H.on_each(qreg))

    if circuit_type == "cirq":

        def convert(circ):
            return circ

    elif circuit_type == "qiskit":
        convert = mitiq_qiskit.to_qiskit
    elif circuit_type == "pyquil":
        convert = mitiq_pyquil.to_pyquil

    circuit = convert(base_circuit)
    measured = pauli.measure_in(circuit)

    qreg = [cirq.NamedQubit(name) for name in names]
    expected = cirq.Circuit(
        # Original circuit.
        base_circuit.all_operations(),
        # Basis rotations.
        xrotation.on(qreg[support[0]]),
        yrotation.on(qreg[support[1]]),
        # Measurements.
        cirq.measure(*[qreg[s] for s in support]),
    )
    if circuit_type == "cirq":
        assert _equal(measured, expected, require_qubit_equality=True)
    else:
        assert measured == convert(expected)


def test_pauli_measure_in_bad_qubits_error():
    n = 5
    pauli = PauliString(spec="X" * n)
    circuit = cirq.Circuit(cirq.H.on_each(cirq.LineQubit.range(n - 1)))

    with pytest.raises(ValueError, match="Qubit mismatch."):
        pauli.measure_in(circuit)


def test_can_be_measured_with_single_qubit():
    pauli = PauliString(spec="Z")

    assert pauli.can_be_measured_with(PauliString(spec="I"))
    assert not pauli.can_be_measured_with(PauliString(spec="X"))
    assert not pauli.can_be_measured_with(PauliString(spec="Y"))
    assert pauli.can_be_measured_with(PauliString(spec="Z", coeff=-0.5))
    assert pauli.can_be_measured_with(pauli)


def test_can_be_measured_with_two_qubits():
    pauli = PauliString(spec="ZX")

    assert pauli.can_be_measured_with(PauliString(spec="Z"))
    assert pauli.can_be_measured_with(PauliString(spec="X", support=(1,)))
    assert not pauli.can_be_measured_with(PauliString(spec="X"))
    assert not pauli.can_be_measured_with(PauliString(spec="Y", coeff=-1.0))

    assert pauli.can_be_measured_with(PauliString(spec="ZX", coeff=0.5))
    assert not pauli.can_be_measured_with(PauliString(spec="ZZ"))


def test_can_be_measured_with_non_overlapping_paulis():
    pauli = PauliString(spec="ZX")

    assert pauli.can_be_measured_with(PauliString(spec="IIZI"))
    assert pauli.can_be_measured_with(PauliString(spec="IIIX"))
    assert pauli.can_be_measured_with(PauliString(spec="IIYZ"))


def test_weight():
    assert PauliString(spec="I").weight() == 0
    assert PauliString(spec="Z").weight() == 1

    n = 4
    assert PauliString(spec="X" * n).weight() == n
    assert PauliString(spec="IX" * n).weight() == n
    assert PauliString(spec="ZX" * n).weight() == 2 * n


def test_observable():
    pauli1 = PauliString(spec="XI", coeff=-1.0)
    pauli2 = PauliString(spec="IZ", coeff=2.0)
    obs = Observable(pauli1, pauli2)

    assert obs.nqubits == 2
    assert obs.qubit_indices == [0, 1]
    assert obs.nterms == 2

    imat = np.identity(2)
    xmat = cirq.unitary(cirq.X)
    zmat = cirq.unitary(cirq.Z)
    correct_matrix = -1.0 * np.kron(xmat, imat) + 2.0 * np.kron(imat, zmat)
    assert np.allclose(obs.matrix(), correct_matrix)


def test_observable_partition_one_set():
    pauli1 = PauliString(spec="ZI")
    pauli2 = PauliString(spec="IZ")
    pauli3 = PauliString(spec="ZZ")
    obs = Observable(pauli1, pauli2, pauli3)
    assert obs.nterms == 3

    sets = obs.partition()
    # expected = {frozenset([pauli1, pauli2, pauli3])}

    assert len(sets) == 1
    # assert sets == expected


def test_observable_partition_single_qubit_paulis():
    x = PauliString(spec="X")
    y = PauliString(spec="Y")
    z = PauliString(spec="Z")
    obs = Observable(x, y, z)
    assert obs.nterms == 3

    sets = obs.partition()
    # expected = {frozenset([p]) for p in (x, y, z)}

    assert len(sets) == 3
    # assert sets == expected


def test_observable_partition_can_be_measured_with():
    n = 10
    nterms = 50
    rng = np.random.RandomState(seed=1)
    obs = Observable(
        *[
            PauliString(
                spec=rng.choice(
                    ("I", "X", "Y", "Z"),
                    size=n,
                    replace=True,
                    p=(0.7, 0.1, 0.1, 0.1),
                )
            )
            for _ in range(nterms)
        ]
    )
    assert obs.nterms == nterms

    for pset in obs.partition():
        pset = list(pset)
        for i in range(len(pset) - 1):
            for j in range(i, len(pset)):
                assert pset[i].can_be_measured_with(pset[j])


def test_observable_measure_in_needs_one_circuit_z():
    pauli1 = PauliString(spec="ZI")
    pauli2 = PauliString(spec="IZ")
    pauli3 = PauliString(spec="ZZ")
    obs = Observable(pauli1, pauli2, pauli3)

    qubits = cirq.LineQubit.range(2)
    circuit = cirq.testing.random_circuit(qubits, 3, 1, random_state=1)

    measures_obs_circuits = obs._measure_in(circuit)
    assert len(measures_obs_circuits) == 1

    expected = circuit + cirq.measure(*qubits)
    assert _equal(
        measures_obs_circuits[0],
        expected,
        require_qubit_equality=True,
        require_measurement_equality=True,
    )


def test_observable_measure_in_needs_one_circuit_x():
    pauli1 = PauliString(spec="XI")
    pauli2 = PauliString(spec="IX")
    pauli3 = PauliString(spec="XX")
    obs = Observable(pauli1, pauli2, pauli3)

    qubits = cirq.LineQubit.range(2)
    circuit = cirq.testing.random_circuit(qubits, 3, 1, random_state=1)

    measures_obs_circuits = obs._measure_in(circuit)
    assert len(measures_obs_circuits) == 1

    expected = circuit + xrotation.on_each(*qubits) + cirq.measure(*qubits)
    assert _equal(
        measures_obs_circuits[0],
        expected,
        require_qubit_equality=True,
        require_measurement_equality=True,
    )


def test_observable_measure_in_needs_two_circuits():
    obs = Observable(PauliString(spec="X"), PauliString(spec="Z"))

    q = cirq.LineQubit(0)
    circuit = cirq.Circuit(cirq.H.on(q))

    measures_obs_circuits = sorted(obs._measure_in(circuit), key=len)
    assert len(measures_obs_circuits) == 2

    expected_circuits = [
        circuit + cirq.measure(q),
        circuit + xrotation.on(q) + cirq.measure(q),
    ]
    for expected, measured in zip(expected_circuits, measures_obs_circuits):
        assert _equal(
            measured,
            expected,
            require_qubit_equality=True,
            require_measurement_equality=True,
        )
