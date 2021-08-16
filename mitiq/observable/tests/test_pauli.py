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

from mitiq.interface import convert_to_mitiq, mitiq_qiskit, mitiq_pyquil
from mitiq.observable import PauliString
from mitiq.rem import MeasurementResult
from mitiq.utils import _equal


# Basis rotations to measure Pauli X and Y.
xrotation = cirq.SingleQubitCliffordGate.Y_nsqrt
yrotation = cirq.SingleQubitCliffordGate.X_sqrt

# Matrices.
imat = np.identity(2)
xmat = cirq.unitary(cirq.X)
zmat = cirq.unitary(cirq.Z)


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


def test_pauli_eq():
    assert PauliString(spec="Z") == PauliString(spec="Z")
    assert PauliString(spec="X") != PauliString(spec="Z")
    assert PauliString(spec="Z") != PauliString(spec="Z", coeff=-1.0)

    assert PauliString(spec="Z", support=(0,)) != PauliString(
        spec="Z", support=(1,)
    )
    assert PauliString(spec="IZ") == PauliString(spec="Z", support=(1,))
    assert PauliString(spec="XY") == PauliString(spec="YX", support=(1, 0))

    assert {PauliString(spec="Z"), PauliString(spec="Z")} == {
        PauliString(spec="Z")
    }


def test_matrix():
    assert np.allclose(PauliString(spec="X").matrix(), xmat)
    assert np.allclose(PauliString(spec="Z", coeff=-0.5).matrix(), -0.5 * zmat)
    assert np.allclose(PauliString(spec="ZZ").matrix(), np.kron(zmat, zmat))
    assert np.allclose(PauliString(spec="XZ").matrix(), np.kron(xmat, zmat))


def test_pauli_matrix_include_qubits():
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


# Note: For testing `PauliString._expectation_from_measurements`, it makes no
# difference whether the Pauli is X, Y, or Z. This is because we assume
# measurements are obtained by single-qubit basis rotations. So it only matters
# whether the Pauli is I (identity) or X, Y, Z (not identity). For consistency
# we just use Z for not identity below.


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("nqubits", [1, 2, 5])
def test_expectation_from_measurements_identity(seed, nqubits):
    """For P = cI, asserts ⟨P⟩ = c."""
    rng = np.random.RandomState(seed)
    coeff = rng.random()
    pauli = PauliString(spec="I", coeff=coeff)

    measurements = MeasurementResult(
        rng.randint(low=0, high=1 + 1, size=(100, nqubits)).tolist()
    )
    assert np.isclose(
        pauli._expectation_from_measurements(measurements), coeff,
    )


def test_expectation_from_measurements_two_qubits():
    measurements = MeasurementResult([[0, 1] * 1_000])

    z0 = PauliString(spec="Z", support=(0,))
    assert np.isclose(z0._expectation_from_measurements(measurements), 1.0,)
    zi = PauliString(spec="ZI")
    assert np.isclose(zi._expectation_from_measurements(measurements), 1.0,)

    z1 = PauliString(spec="Z", support=(1,))
    assert np.isclose(z1._expectation_from_measurements(measurements), -1.0,)
    iz = PauliString(spec="IZ")
    assert np.isclose(iz._expectation_from_measurements(measurements), -1.0,)

    zz = PauliString(spec="ZZ")
    assert np.isclose(zz._expectation_from_measurements(measurements), -1.0,)
