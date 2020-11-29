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

"""Unit tests for Collector and CircuitCollection."""
import pytest
from typing import List

import numpy as np

import cirq
import pyquil

from mitiq.collector import CircuitCollection, Collector
from mitiq.utils import _equal


def executor_batched(circuits) -> np.ndarray:
    return np.zeros(len(circuits))


def executor_serial_typed(*args) -> float:
    return 0.0


def executor_serial(*args):
    return 0.0


def executor_pyquil_batched(programs) -> List[float]:
    for p in programs:
        if not isinstance(p, pyquil.Program):
            raise TypeError

    return [0.0] * len(programs)


def test_collector_simple():
    collector = Collector(executor=executor_batched, max_batch_size=10)
    assert collector.can_batch
    assert collector._max_batch_size == 10
    assert collector.calls_to_executor == 0


def test_collector_is_batched_executor():
    assert Collector.is_batched_executor(executor=executor_batched)
    assert not Collector.is_batched_executor(executor=executor_serial_typed)
    assert not Collector.is_batched_executor(executor=executor_serial)


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

    assert np.allclose(results, np.zeros(len(circuits)))
    assert collector.calls_to_executor == np.ceil(ncircuits / batch_size)


def test_circuit_collection_simple():
    qbit = cirq.LineQubit(0)
    hcircuits = [cirq.Circuit(cirq.H(qbit)) for _ in range(3)]
    xcircuits = [cirq.Circuit(cirq.X(qbit)) for _ in range(2)]
    collection = CircuitCollection(circuits=hcircuits + xcircuits)

    assert len(collection.all) == 5
    assert len(collection.unique) == 2

    assert hcircuits[0] in collection
    assert xcircuits[0] in collection
    assert cirq.Circuit(cirq.Z(qbit)) not in collection


def test_circuit_collection_pyquil():
    collection = CircuitCollection([pyquil.Program(pyquil.gates.X(0))])

    assert len(collection.all) == 1
    assert len(collection.unique) == 1
    assert collection._counts == {0: 1}
    for p in collection.unique:
        assert isinstance(p, pyquil.Program)


def test_circuit_collection_simple_pyquil():
    hcircuits = [pyquil.Program(pyquil.gates.H(0)) for _ in range(3)]
    xcircuits = [pyquil.Program(pyquil.gates.X(0)) for _ in range(3)]
    circuits = CircuitCollection(circuits=hcircuits + xcircuits)

    assert len(circuits.all) == 6
    assert len(circuits.unique) == 2

    assert hcircuits[0] in circuits
    assert xcircuits[0] in circuits
    assert pyquil.Program(pyquil.gates.Z(0)) not in circuits


def test_circuit_collection_staggered_two_circuits():
    qbit = cirq.LineQubit(0)
    circuits = [cirq.Circuit(cirq.X(qbit)), cirq.Circuit(cirq.Z(qbit))] * 10

    collection = CircuitCollection(circuits)
    assert collection._counts == {0: 10, 1: 10}


def test_circuit_collection_counts():
    qbit = cirq.LineQubit(0)
    circuits = [
        cirq.Circuit(cirq.X(qbit)),
        cirq.Circuit(cirq.Z(qbit)),
        cirq.Circuit(cirq.Y(qbit)),
        cirq.Circuit(cirq.Y(qbit)),
        cirq.Circuit(cirq.X(qbit)),
        cirq.Circuit(cirq.Y(qbit)),
        cirq.Circuit(cirq.H(qbit)),
        cirq.Circuit(cirq.Y(qbit)),
        cirq.Circuit(cirq.H(qbit)),
    ]
    collection = CircuitCollection(circuits)
    assert len(collection.unique) == 4
    assert collection._counts == {0: 2, 1: 1, 2: 4, 6: 2}


def test_circuit_collection_unique_with_counts():
    qbit = cirq.LineQubit(0)
    circuits = [
        cirq.Circuit(cirq.X(qbit)),
        cirq.Circuit(cirq.X(qbit)),
        cirq.Circuit(cirq.Y(qbit)),
        cirq.Circuit(cirq.Z(qbit)),
    ]
    collection = CircuitCollection(circuits)
    assert collection._counts == {0: 2, 2: 1, 3: 1}

    unique_with_counts = collection.unique_with_counts()

    correct = [
        (cirq.Circuit(cirq.X(qbit)), 2),
        (cirq.Circuit(cirq.Y(qbit)), 1),
        (cirq.Circuit(cirq.Z(qbit)), 1),
    ]

    for i in range(len(unique_with_counts)):
        assert _equal(unique_with_counts[i][0], correct[i][0])
        assert unique_with_counts[i][1] == correct[i][1]


def test_circuit_collection_multiplicity_of():
    rng = np.random.RandomState(1)

    qbit = cirq.LineQubit(0)
    circuits = [
        cirq.Circuit(cirq.X(qbit)),
        cirq.Circuit(cirq.Z(qbit)),
        cirq.Circuit(cirq.Y(qbit)),
        cirq.Circuit(cirq.Y(qbit)),
        cirq.Circuit(cirq.X(qbit)),
        cirq.Circuit(cirq.Y(qbit)),
        cirq.Circuit(cirq.H(qbit)),
        cirq.Circuit(cirq.Y(qbit)),
        cirq.Circuit(cirq.H(qbit)),
    ]

    for _ in range(10):
        rng.shuffle(circuits)
        collection = CircuitCollection(circuits)

        assert len(collection.unique) == 4
        assert collection.multiplicity_of(cirq.Circuit(cirq.X(qbit))) == 2
        assert collection.multiplicity_of(cirq.Circuit(cirq.Z(qbit))) == 1
        assert collection.multiplicity_of(cirq.Circuit(cirq.Y(qbit))) == 4
        assert collection.multiplicity_of(cirq.Circuit(cirq.H(qbit))) == 2
        assert collection.multiplicity_of(cirq.Circuit(cirq.I(qbit))) == 0


@pytest.mark.parametrize("item", ("circuit", 1, None))
def test_circuit_collection_multiplicity_of_bad_type(item):
    qbit = cirq.LineQubit(0)
    collection = CircuitCollection([cirq.Circuit(cirq.X(qbit))])
    assert collection.multiplicity_of(item) == 0
