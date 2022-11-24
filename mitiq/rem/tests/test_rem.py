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

"""Unit tests for readout confusion inversion."""
from cmath import isclose
from functools import partial
import cirq
from cirq.experiments.single_qubit_readout_calibration_test import (
    NoisySingleQubitReadoutSampler,
)
import numpy as np
import pytest

from mitiq.observable.observable import Observable
from mitiq.observable.pauli import PauliString
from mitiq._typing import MeasurementResult
from mitiq.rem.inverse_confusion_matrix import (
    generate_inverse_confusion_matrix,
)
from mitiq.rem.rem import execute_with_rem, mitigate_executor, rem_decorator
from mitiq.raw import execute as raw_execute
from mitiq.interface.mitiq_cirq import sample_bitstrings

# Default qubit register and circuit for unit tests
qreg = [cirq.LineQubit(i) for i in range(2)]
circ = cirq.Circuit(cirq.ops.X.on_each(*qreg), cirq.measure_each(*qreg))
observable = Observable(PauliString("ZI"), PauliString("IZ"))


def noisy_readout_executor(
    circuit, p0: float = 0.01, p1: float = 0.01, shots: int = 8192
) -> MeasurementResult:
    simulator = NoisySingleQubitReadoutSampler(p0, p1)
    result = simulator.run(circuit, repetitions=shots)

    return MeasurementResult(
        result=np.column_stack(list(result.measurements.values())),
        qubit_indices=tuple(
            # q[2:-1] is necessary to convert "q(number)" into "number"
            int(q[2:-1])
            for k in result.measurements.keys()
            for q in k.split(",")
        ),
    )


npX = np.array([[0, 1], [1, 0]])
"""Defines the sigma_x Pauli matrix in SU(2) algebra as a (2,2) `np.array`."""

npZ = np.array([[1, 0], [0, -1]])
"""Defines the sigma_z Pauli matrix in SU(2) algebra as a (2,2) `np.array`."""


# An invalid executor for unit tests since it returns an expectation
def invalid_executor(circuit) -> float:
    wavefunction = circuit.final_state_vector(
        ignore_terminal_measurements=True
    )
    return np.real(wavefunction.conj().T @ np.kron(npX, npZ) @ wavefunction)


def test_rem_invalid_executor():
    identity = np.identity(4)
    with pytest.raises(TypeError, match="not of type MeasurementResult"):
        execute_with_rem(
            circ,
            invalid_executor,
            observable,
            inverse_confusion_matrix=identity,
        )


def test_rem_identity():
    executor = partial(sample_bitstrings, noise_level=(0,))
    identity = np.identity(4)
    result = execute_with_rem(
        circ, executor, observable, inverse_confusion_matrix=identity
    )
    assert np.isclose(result, -2.0)


def test_rem_with_matrix():
    # test with an executor that completely flips results
    p0 = 1
    p1 = 1
    noisy_executor = partial(noisy_readout_executor, p0=p0, p1=p1)
    unmitigated = raw_execute(circ, noisy_executor, observable)
    assert np.isclose(unmitigated, 2.0)

    inverse_confusion_matrix = np.array(
        [
            [0, 0, 0, 1],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [1, 0, 0, 0],
        ]
    )

    mitigated = execute_with_rem(
        circ,
        noisy_executor,
        observable,
        inverse_confusion_matrix=inverse_confusion_matrix,
    )
    assert np.isclose(mitigated, -2.0)


def test_rem_with_invalid_matrix():
    executor = partial(sample_bitstrings, noise_level=(0,))
    identity = np.identity(2)
    with pytest.raises(ValueError):
        execute_with_rem(
            circ, executor, observable, inverse_confusion_matrix=identity
        )


def test_doc_is_preserved():
    """Tests that the doc of the original executor is preserved."""

    def first_executor(circuit):
        """Doc of the original executor."""
        return 0

    identity = np.identity(4)

    mit_executor = mitigate_executor(
        first_executor, observable, inverse_confusion_matrix=identity
    )
    assert mit_executor.__doc__ == first_executor.__doc__

    @rem_decorator(observable, inverse_confusion_matrix=identity)
    def second_executor(circuit):
        """Doc of the original executor."""
        return 0

    assert second_executor.__doc__ == first_executor.__doc__


@pytest.mark.parametrize(
    "p0, p1, atol",
    [
        (0, 0, 0),
        (1, 1, 0),
        (
            1,
            0,
            2.1,
        ),  # catastrophic failure from all 0s state, which is why atol=2.1
        (
            0,
            1,
            0,
        ),  # undetectable: results in all 1s state, which is the desired state
        (0.3, 0.1, 0),
        (0.1, 0.3, 0),
        (0.001, 0.001, 1e-3),
        (0.02, 0.04, 0),
        (0.3, 0.7, 0),
    ],
)
def test_mitigate_executor(p0, p1, atol):
    true_rem_value = -2.0

    noisy_executor = partial(noisy_readout_executor, p0=p0, p1=p1)

    num_qubits = len(qreg)
    inverse_confusion_matrix = generate_inverse_confusion_matrix(
        num_qubits, p0=p0, p1=p1
    )

    base = raw_execute(circ, noisy_executor, observable)

    mitigated_executor = mitigate_executor(
        noisy_executor,
        observable,
        inverse_confusion_matrix=inverse_confusion_matrix,
    )
    rem_value = mitigated_executor(circ)
    assert abs(true_rem_value - rem_value) <= abs(
        true_rem_value - base
    ) or isclose(
        abs(true_rem_value - rem_value),
        abs(true_rem_value - base),
        abs_tol=atol,
    )


def test_rem_decorator():
    true_rem_value = -2.0

    num_qubits = len(qreg)

    # test with an executor that completely flips results
    p0 = 1
    p1 = 1
    inverse_confusion_matrix = generate_inverse_confusion_matrix(
        num_qubits, p0=p0, p1=p1
    )

    @rem_decorator(
        observable, inverse_confusion_matrix=inverse_confusion_matrix
    )
    def noisy_readout_decorated_executor(qp) -> MeasurementResult:
        return noisy_readout_executor(qp, p0=p0, p1=p1)

    noisy_executor = partial(noisy_readout_executor, p0=p0, p1=p1)

    base = raw_execute(circ, noisy_executor, observable)

    rem_value = noisy_readout_decorated_executor(circ)
    assert abs(true_rem_value - rem_value) < abs(true_rem_value - base)
