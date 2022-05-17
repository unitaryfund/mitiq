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
from functools import partial
import cirq
from cirq.experiments.single_qubit_readout_calibration_test import (
    NoisySingleQubitReadoutSampler,
)
import numpy as np

from mitiq import Observable, Executor
from mitiq.observable.pauli import PauliString
from mitiq.rem.measurement_result import MeasurementResult
from mitiq.rem.rci import execute_with_rci
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
            int(q) for k in result.measurements.keys() for q in k.split(",")
        ),
    )


def test_rci_identity():
    executor = partial(sample_bitstrings, noise_level=(0,))
    identity = np.identity(4)
    result = execute_with_rci(
        circ, executor, observable, inverse_confusion_matrix=identity
    )
    assert np.isclose(result, -2.0)


def test_rci_without_matrix():
    # test with an executor that completely flips results
    p0 = 1
    p1 = 1
    noisy_executor = partial(noisy_readout_executor, p0=p0, p1=p1)
    executor = Executor(noisy_executor)
    unmitigated = executor.evaluate([circ], observable)
    assert np.isclose(unmitigated, 2.0)

    mitigated = execute_with_rci(circ, executor, observable, p0=p0, p1=p1)
    assert np.isclose(mitigated, -2.0)


def test_rci_with_matrix():
    # test with an executor that completely flips results
    p0 = 1
    p1 = 1
    noisy_executor = partial(noisy_readout_executor, p0=p0, p1=p1)
    executor = Executor(noisy_executor)
    unmitigated = executor.evaluate([circ], observable)
    assert np.isclose(unmitigated, 2.0)

    inverse_confusion_matrix = [
        [0, 0, 0, 1],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [1, 0, 0, 0],
    ]

    mitigated = execute_with_rci(
        circ,
        executor,
        observable,
        inverse_confusion_matrix=inverse_confusion_matrix,
    )
    assert np.isclose(mitigated, -2.0)


if __name__ == "__main__":
    # test_rci_identity()
    # test_rci_without_matrix()
    test_rci_with_matrix()

