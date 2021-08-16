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

import numpy as np
import cirq

from mitiq.observable.observable import Observable
from mitiq.observable.pauli import PauliString, PauliStringSet
from mitiq.rem.measurement_result import MeasurementResult
from mitiq.utils import _equal


# Basis rotations to measure Pauli X and Y.
xrotation = cirq.SingleQubitCliffordGate.Y_nsqrt
yrotation = cirq.SingleQubitCliffordGate.X_sqrt

# Matrices.
imat = np.identity(2)
xmat = cirq.unitary(cirq.X)
zmat = cirq.unitary(cirq.Z)


def test_observable():
    pauli1 = PauliString(spec="XI", coeff=-1.0)
    pauli2 = PauliString(spec="IZ", coeff=2.0)
    obs = Observable(pauli1, pauli2)

    assert obs.nqubits == 2
    assert obs.qubit_indices == [0, 1]
    assert obs.nterms == 2
    assert obs.ngroups == 1

    correct_matrix = -1.0 * np.kron(xmat, imat) + 2.0 * np.kron(imat, zmat)
    assert np.allclose(obs.matrix(), correct_matrix)


def test_observable_partition_one_set():
    pauli1 = PauliString(spec="ZI")
    pauli2 = PauliString(spec="IZ")
    pauli3 = PauliString(spec="ZZ")
    obs = Observable(pauli1, pauli2, pauli3)
    assert obs.nterms == 3

    assert obs.ngroups == 1
    assert obs.groups[0] == PauliStringSet(pauli1, pauli2, pauli3)


def test_observable_partition_single_qubit_paulis():
    x = PauliString(spec="X")
    y = PauliString(spec="Y")
    z = PauliString(spec="Z")
    obs = Observable(x, y, z)
    assert obs.nterms == 3

    obs.partition(seed=2)
    assert obs.groups == [PauliStringSet(x), PauliStringSet(y), PauliStringSet(z)]


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

    assert obs.nqubits == n
    assert obs.nterms == nterms
    assert obs.ngroups <= nterms

    for pset in obs.groups:
        pauli_list = list(pset.elements)
        for i in range(len(pauli_list) - 1):
            for j in range(i, len(pauli_list)):
                assert pauli_list[i].can_be_measured_with(pauli_list[j])


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


def test_observable_expectation_from_measurements_one_pauli_string():
    obs = Observable(PauliString(spec="Z"))

    measurements = MeasurementResult([[0], [0], [0], [0], [0], [0], [0], [0], [0], [0]])
    expectation = obs._expectation_from_measurements([measurements])
    assert np.isclose(expectation, 1.0)

    measurements = MeasurementResult([[1], [1], [1], [1], [1], [1], [1], [1], [1], [1]])
    expectation = obs._expectation_from_measurements([measurements])
    assert np.isclose(expectation, -1.0)

    measurements = MeasurementResult([[0], [1], [0], [1], [0], [1], [0], [1], [0], [1]])
    expectation = obs._expectation_from_measurements([measurements])
    assert np.isclose(expectation, 0.0)


def test_observable_expectation_from_measurements_two_pauli_strings():
    obs = Observable(PauliString(spec="Z", coeff=2.5), PauliString(spec="X"))

    bits = MeasurementResult([[0], [0], [0], [0], [0], [0], [0], [0], [0], [0]])
    expectation = obs._expectation_from_measurements([bits, bits])
    assert np.isclose(expectation, 3.5)

#
# def test_observable_expectation():
#     obs = Observable(PauliString(spec="Z"), PauliString(spec="Z"))
#     assert obs.ngroups == 1
#
#     q = cirq.LineQubit(0)
#     circuit = cirq.Circuit(cirq.I(q))
#
#     def execute(circuit: cirq.Circuit) -> List[List[int]]:
#         return (
#             cirq.Simulator().run(circuit, repetitions=1000).measurements["z"]
#         )
#
#     assert np.isclose(obs.expectation(circuit, executor=execute), 2.0)
