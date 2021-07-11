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
import pytest
from typing import List

import numpy as np

import cirq
import pyquil

from mitiq.collector import Collector, generate_collected_executor


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
def executor_serial_measurements(circuit) -> cirq.Result:
    return cirq.Simulator().run(circuit, repetitions=10)


def executor_batched_measurements(circuits) -> List[cirq.Result]:
    return [
        cirq.Simulator().run(circuit, repetitions=10) for circuit in circuits
    ]


def test_collector_simple():
    collector = Collector(executor=executor_batched, max_batch_size=10)
    assert collector.can_batch
    assert collector._max_batch_size == 10
    assert collector.calls_to_executor == 0


def test_collector_is_batched_executor():
    assert Collector.is_batched_executor(executor_batched)
    assert not Collector.is_batched_executor(executor_serial_typed)
    assert not Collector.is_batched_executor(executor_serial)
    assert not Collector.is_batched_executor(executor_serial_measurements)
    assert Collector.is_batched_executor(executor_batched_measurements)


@pytest.mark.parametrize("ncircuits", (5, 10, 25))
@pytest.mark.parametrize("executor", (executor_batched, executor_serial))
def test_run_collector_identical_circuits_batched(ncircuits, executor):
    collector = Collector(executor=executor, max_batch_size=10)
    circuits = [cirq.Circuit(cirq.H(cirq.LineQubit(0)))] * ncircuits
    results = collector.run(circuits)

    assert np.allclose(results, np.zeros(ncircuits))
    assert collector.calls_to_executor == 1


@pytest.mark.parametrize("batch_size", (1, 2, 10))
def test_run_collector_nonidentical_pyquil_programs(batch_size):
    collector = Collector(
        executor=executor_pyquil_batched, max_batch_size=batch_size
    )
    assert collector.can_batch

    circuits = [
        pyquil.Program(pyquil.gates.X(0)),
        pyquil.Program(pyquil.gates.H(0)),
    ] * 10
    results = collector.run(circuits)

    assert np.allclose(results, np.zeros(len(circuits)))
    if batch_size == 1:
        assert collector.calls_to_executor == 2
    else:
        assert collector.calls_to_executor == 1


@pytest.mark.parametrize("ncircuits", (10, 11, 23))
@pytest.mark.parametrize("batch_size", (1, 2, 5, 50))
def test_run_collector_all_unique(ncircuits, batch_size):
    collector = Collector(executor=executor_batched, max_batch_size=batch_size)
    assert collector.can_batch

    random_state = np.random.RandomState(seed=1)
    circuits = [
        cirq.testing.random_circuit(
            qubits=4, n_moments=10, op_density=1, random_state=random_state
        )
        for _ in range(ncircuits)
    ]
    results = collector.run(circuits)

    assert np.allclose(results, np.zeros(ncircuits))
    assert collector.calls_to_executor == np.ceil(ncircuits / batch_size)


@pytest.mark.parametrize("ncircuits", (5, 21))
@pytest.mark.parametrize("force_run_all", (True, False))
def test_run_collector_force_run_all_serial_executor_identical_circuits(
    ncircuits, force_run_all
):
    collector = Collector(executor=executor_serial)
    assert not collector.can_batch

    circuits = [cirq.Circuit(cirq.H(cirq.LineQubit(0)))] * ncircuits
    results = collector.run(circuits, force_run_all=force_run_all)

    assert np.allclose(results, np.zeros(ncircuits))
    if force_run_all:
        assert collector.calls_to_executor == ncircuits
    else:
        assert collector.calls_to_executor == 1


@pytest.mark.parametrize("s", (50, 100, 150))
@pytest.mark.parametrize("b", (1, 2, 100))
def test_run_collector_preserves_order(s, b):
    rng = np.random.RandomState(1)

    collector = Collector(executor=executor_batched_unique, max_batch_size=b)
    assert collector.can_batch

    circuits = [
        cirq.Circuit(cirq.H(cirq.LineQubit(0))),
        cirq.Circuit([cirq.H(cirq.LineQubit(0))] * 2),
    ]
    batch = [circuits[i] for i in rng.random_integers(low=0, high=1, size=s)]

    assert np.allclose(collector.run(batch), executor_batched_unique(batch))


@pytest.mark.parametrize("executor", (executor_serial, executor_batched))
@pytest.mark.parametrize("ncircuits", (10, 25))
@pytest.mark.parametrize("rval", (0.0, 1.0))
def test_generate_collected_executor(executor, ncircuits, rval):
    collected_executor = generate_collected_executor(
        executor, return_value=rval
    )
    expvals = collected_executor([cirq.Circuit()] * ncircuits)
    assert np.allclose(expvals, np.full(shape=(ncircuits,), fill_value=rval))


def test_generate_collected_executor_not_callable():
    with pytest.raises(ValueError, match="must be callable"):
        generate_collected_executor(None)
