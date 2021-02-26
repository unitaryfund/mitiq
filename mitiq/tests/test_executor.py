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

from typing import List
import pytest

import numpy as np

import cirq
from mitiq.executor import Executor, MeasurementResult


def execute_serial_untyped(x):
    return x ** 2


def executor_serial_typed(x) -> float:
    return x ** 2


def execute_batch(xvals) -> List[float]:
    return [x ** 2 for x in xvals]


@pytest.mark.parametrize(
    "execute", (execute_serial_untyped, executor_serial_typed)
)
def test_executor_with_execute_serial(execute):
    executor = Executor(circuit_to_expectation=execute)
    assert not executor._can_batch()

    result = executor.execute(to_run=1)
    assert result == 1
    assert executor.expectation_history == [1]
    assert executor.last_expectation == 1

    next_result = executor.execute(to_run=2)
    assert next_result == 4
    assert executor.expectation_history == [1, 4]
    assert executor.last_expectation == 4

    assert executor.measurement_history == []
    assert executor.circuit_history == [1, 2]


def test_executor_with_execute_batch():
    executor = Executor(circuit_to_expectation=execute_batch)
    assert executor._can_batch()

    result = executor.execute(to_run=[1, 2])
    assert result == [1, 4]
    assert executor.expectation_history == [[1, 4]]
    assert executor.last_expectation == [1, 4]

    next_result = executor.execute(to_run=[3, 4])
    assert next_result == [9, 16]
    assert executor.expectation_history == [[1, 4], [9, 16]]
    assert executor.last_expectation == [9, 16]

    assert executor.measurement_history == []
    assert executor.circuit_history == [[1, 2], [3, 4]]


def test_executor_input_combinations():
    def generic():
        pass

    with pytest.raises(ValueError):
        Executor(circuit_to_measurement=generic)

    with pytest.raises(ValueError):
        Executor(measurement_to_expectation=generic)

    assert Executor(circuit_to_expectation=generic)
    assert Executor(
        circuit_to_measurement=generic,
        measurement_to_expectation=generic,
    )


def test_executor_with_measurement_result():
    def to_measurement(circuit) -> MeasurementResult:
        return cirq.Simulator().run(circuit, repetitions=1000)

    def ground_state_prob(measurement: MeasurementResult) -> float:
        return np.average(np.sum(measurement.measurements["z"], axis=1) == 0)

    executor = Executor(
        circuit_to_measurement=to_measurement,
        measurement_to_expectation=ground_state_prob,
    )
    assert not executor._can_batch()

    qreg = cirq.LineQubit.range(3)
    circuit1 = cirq.Circuit(cirq.measure(*qreg, key="z"))
    circuit2 = cirq.Circuit(cirq.X.on_each(qreg), cirq.measure(*qreg, key="z"))
    expected_results = [1.0, 0.0]

    for circuit, expected in zip((circuit1, circuit2), expected_results):
        assert executor.execute(circuit) == expected
        assert executor.last_expectation == expected

        meas = executor.last_measurement
        assert ground_state_prob(meas) == executor.last_expectation

    assert executor.expectation_history == expected_results
    assert executor.circuit_history[0] is circuit1
    assert executor.circuit_history[1] is circuit2
