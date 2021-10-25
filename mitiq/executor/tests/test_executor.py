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

"""Unit tests for Collector."""
import functools
import pytest
from typing import List

import numpy as np

import cirq
import pyquil

from mitiq.executor.executor import Executor
from mitiq.rem import MeasurementResult
from mitiq.observable import Observable, PauliString
from mitiq.interface.mitiq_cirq import compute_density_matrix


# Serial / batched executors which return floats.
def executor_batched(circuits, **kwargs) -> List[float]:
    return [
        float(v)
        for v in np.full(
            shape=(len(circuits),),
            fill_value=kwargs.setdefault("return_value", 0.0),
        )
    ]


def executor_batched_unique(circuits) -> List[float]:
    return [executor_serial_unique(circuit) for circuit in circuits]


def executor_serial_unique(circuit):
    return float(len(circuit))


def executor_serial_typed(*args, **kwargs) -> float:
    return executor_serial(*args, **kwargs)


def executor_serial(*args, **kwargs):
    return kwargs.setdefault("return_value", 0.0)


def executor_pyquil_batched(programs) -> List[float]:
    for p in programs:
        if not isinstance(p, pyquil.Program):
            raise TypeError

    return [0.0] * len(programs)


# Serial / batched executors which return measurements.
def executor_serial_measurements(circuit) -> MeasurementResult:
    # Assume there is only one measurement key in the circuit.
    assert len(circuit.all_measurement_keys()) == 1

    key = circuit.all_measurement_keys().pop()
    backend = cirq.Simulator()
    return MeasurementResult(
        backend.run(circuit, repetitions=10).measurements[key].tolist()
    )


def executor_batched_measurements(circuits) -> List[MeasurementResult]:
    return [executor_serial_measurements(circuit) for circuit in circuits]


def test_executor_simple():
    collector = Executor(executor=executor_batched, max_batch_size=10)
    assert collector.can_batch
    assert collector._max_batch_size == 10
    assert collector.calls_to_executor == 0


def test_executor_is_batched_executor():
    assert Executor.is_batched_executor(executor_batched)
    assert not Executor.is_batched_executor(executor_serial_typed)
    assert not Executor.is_batched_executor(executor_serial)
    assert not Executor.is_batched_executor(executor_serial_measurements)
    # assert Collector.is_batched_executor(executor_batched_measurements)


@pytest.mark.parametrize("ncircuits", (5, 10, 25))
@pytest.mark.parametrize("executor", (executor_batched, executor_serial))
def test_run_executor_identical_circuits_batched(ncircuits, executor):
    collector = Executor(executor=executor, max_batch_size=10)
    circuits = [cirq.Circuit(cirq.H(cirq.LineQubit(0)))] * ncircuits
    results = collector._run(circuits)

    assert np.allclose(results, np.zeros(ncircuits))
    assert collector.calls_to_executor == 1


@pytest.mark.parametrize("batch_size", (1, 2, 10))
def test_run_executor_nonidentical_pyquil_programs(batch_size):
    collector = Executor(
        executor=executor_pyquil_batched, max_batch_size=batch_size
    )
    assert collector.can_batch

    circuits = [
        pyquil.Program(pyquil.gates.X(0)),
        pyquil.Program(pyquil.gates.H(0)),
    ] * 10
    results = collector._run(circuits)

    assert np.allclose(results, np.zeros(len(circuits)))
    if batch_size == 1:
        assert collector.calls_to_executor == 2
    else:
        assert collector.calls_to_executor == 1


@pytest.mark.parametrize("ncircuits", (10, 11, 23))
@pytest.mark.parametrize("batch_size", (1, 2, 5, 50))
def test_run_executor_all_unique(ncircuits, batch_size):
    collector = Executor(executor=executor_batched, max_batch_size=batch_size)
    assert collector.can_batch

    random_state = np.random.RandomState(seed=1)
    circuits = [
        cirq.testing.random_circuit(
            qubits=4, n_moments=10, op_density=1, random_state=random_state
        )
        for _ in range(ncircuits)
    ]
    results = collector._run(circuits)

    assert np.allclose(results, np.zeros(ncircuits))
    assert collector.calls_to_executor == np.ceil(ncircuits / batch_size)


@pytest.mark.parametrize("ncircuits", (5, 21))
@pytest.mark.parametrize("force_run_all", (True, False))
def test_run_executor_force_run_all_serial_executor_identical_circuits(
    ncircuits, force_run_all
):
    collector = Executor(executor=executor_serial)
    assert not collector.can_batch

    circuits = [cirq.Circuit(cirq.H(cirq.LineQubit(0)))] * ncircuits
    results = collector._run(circuits, force_run_all=force_run_all)

    assert np.allclose(results, np.zeros(ncircuits))
    if force_run_all:
        assert collector.calls_to_executor == ncircuits
    else:
        assert collector.calls_to_executor == 1


@pytest.mark.parametrize("s", (50, 100, 150))
@pytest.mark.parametrize("b", (1, 2, 100))
def test_run_executor_preserves_order(s, b):
    rng = np.random.RandomState(1)

    collector = Executor(executor=executor_batched_unique, max_batch_size=b)
    assert collector.can_batch

    circuits = [
        cirq.Circuit(cirq.H(cirq.LineQubit(0))),
        cirq.Circuit([cirq.H(cirq.LineQubit(0))] * 2),
    ]
    batch = [circuits[i] for i in rng.random_integers(low=0, high=1, size=s)]

    assert np.allclose(collector._run(batch), executor_batched_unique(batch))


def test_executor_evaluate_density_matrix():
    obs = Observable(PauliString("Z"))

    q = cirq.LineQubit(0)
    circuits = [cirq.Circuit(cirq.I.on(q)), cirq.Circuit(cirq.X.on(q))]

    compute_dm = functools.partial(compute_density_matrix, noise_level=(0,))
    executor = Executor(compute_dm)

    results = executor.evaluate(circuits, obs)
    assert np.allclose(results, [1, -1])
    assert executor.executed_circuits == circuits
    assert np.allclose(executor.quantum_results[0], compute_dm(circuits[0]))
    assert np.allclose(executor.quantum_results[1], compute_dm(circuits[1]))
    assert len(executor.quantum_results) == len(circuits)


def test_executor_evaluate_bitstrings():
    from mitiq.interface.mitiq_cirq import sample_bitstrings

    obs = Observable(PauliString("Z"))

    q = cirq.LineQubit(0)
    circuits = [cirq.Circuit(cirq.I.on(q)), cirq.Circuit(cirq.X.on(q))]

    sample_bitstrings = functools.partial(sample_bitstrings, noise_level=(0,))
    executor = Executor(sample_bitstrings)

    results = executor.evaluate(circuits, obs)
    assert np.allclose(results, [1, -1])
    assert executor.executed_circuits[0] == circuits[0] + cirq.measure(q)
    assert executor.executed_circuits[1] == circuits[1] + cirq.measure(q)
    assert executor.quantum_results[0] == sample_bitstrings(circuits[0] + cirq.measure(q))
    assert executor.quantum_results[1] == sample_bitstrings(circuits[1] + cirq.measure(q))
    assert len(executor.quantum_results) == len(circuits)
