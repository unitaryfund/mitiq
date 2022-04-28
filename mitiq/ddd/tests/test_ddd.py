# Copyright (C) 2022 Unitary Fund
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

"""Unit tests for high-level DDD tools."""

import numpy as np
from pytest import mark
import cirq

from mitiq import QPROGRAM, SUPPORTED_PROGRAM_TYPES, Executor
from mitiq.interface import convert_to_mitiq, convert_from_mitiq
from mitiq.interface.mitiq_cirq import compute_density_matrix

from mitiq.ddd.rules import xx, yy, xyxy
from mitiq.ddd import execute_with_ddd
from mitiq.pec.tests.test_pec import (
    serial_executor,
    batched_executor,
    noiseless_serial_executor,
)

circuit_cirq_a = cirq.Circuit(
    cirq.SWAP(q, q + 1) for q in cirq.LineQubit.range(7)
)

circuit_cirq_b = cirq.Circuit(
    cirq.CNOT(q, q + 1) for q in cirq.LineQubit.range(4)
)
circuit_cirq_b += cirq.inverse(circuit_cirq_b)


def amp_damp_executor(circuit: QPROGRAM, noise: float = 0.1) -> float:
    circuit, _ = convert_to_mitiq(circuit)
    return compute_density_matrix(
        circuit, noise_model=cirq.amplitude_damp, noise_level=(noise,)
    )[0, 0].real


@mark.parametrize("circuit_type", SUPPORTED_PROGRAM_TYPES.keys())
@mark.parametrize("circuit", [circuit_cirq_a, circuit_cirq_b])
@mark.parametrize("rule", [xx, yy, xyxy])
def test_execute_with_ddd_without_noise(circuit_type, circuit, rule):
    """Tests that execute_with_ddd preserves expected results
    in the absence of noise.
    """
    circuit = convert_from_mitiq(circuit, circuit_type)
    true_noiseless_value = 1.0
    unmitigated = noiseless_serial_executor(circuit)
    mitigated = execute_with_ddd(
        circuit,
        executor=noiseless_serial_executor,
        rule=rule,
    )
    error_unmitigated = abs(unmitigated - true_noiseless_value)
    error_mitigated = abs(mitigated - true_noiseless_value)
    assert np.isclose(error_unmitigated, error_mitigated)


@mark.parametrize("circuit_type", SUPPORTED_PROGRAM_TYPES.keys())
@mark.parametrize("circuit", [circuit_cirq_a, circuit_cirq_b])
@mark.parametrize("executor", [serial_executor, batched_executor])
@mark.parametrize("rule", [xx, yy, xyxy])
def test_execute_with_ddd_and_depolarizing_noise(
    circuit_type, circuit, executor, rule
):
    """Tests that with execute_with_ddd the error of a noisy
    expectation value is unchanged with depolarizing noise.
    """
    circuit = convert_from_mitiq(circuit, circuit_type)
    true_noiseless_value = 1.0
    unmitigated = serial_executor(circuit)
    mitigated = execute_with_ddd(
        circuit,
        executor,
        rule=rule,
    )
    error_unmitigated = abs(unmitigated - true_noiseless_value)
    error_mitigated = abs(mitigated - true_noiseless_value)

    # For moment-based depolarizing noise DDD should
    # have no effect (since noise commutes with DDD gates).
    assert np.isclose(error_mitigated, error_unmitigated)


@mark.parametrize("circuit_type", SUPPORTED_PROGRAM_TYPES.keys())
@mark.parametrize("rule", [xx, yy, xyxy])
def test_execute_with_ddd_and_damping_noise(circuit_type, rule):
    """Tests that with execute_with_ddd the error of a noisy
    expectation value is unchanged with depolarizing noise.
    """
    circuit = convert_from_mitiq(circuit_cirq_a, circuit_type)
    true_noiseless_value = 1.0
    unmitigated = amp_damp_executor(circuit)
    mitigated = execute_with_ddd(
        circuit,
        amp_damp_executor,
        rule=rule,
    )
    error_unmitigated = abs(unmitigated - true_noiseless_value)
    error_mitigated = abs(mitigated - true_noiseless_value)

    # For moment-based amplitude-damping noise DDD should
    # have an non-trivial effect (positive or negative).

    # TODO: uncomment the following line and remove the second one
    # after insert_ddd_sequences is implemented.
    # assert not np.isclose(error_mitigated, error_unmitigated)
    assert np.isclose(error_mitigated, error_unmitigated)


@mark.parametrize("executor", [serial_executor, batched_executor])
def test_execute_with_ddd_with_num_trials(executor):
    """Tests the option num_trials of execute_with_ddd."""
    executor = Executor(executor)
    mitigated_1 = execute_with_ddd(
        circuit_cirq_a,
        executor,
        rule=xx,
        num_trials=1,
    )
    assert executor.calls_to_executor == 1
    assert len(executor.executed_circuits) == 1

    mitigated_2 = execute_with_ddd(
        circuit_cirq_a,
        executor,
        rule=xx,
        num_trials=2,
    )
    # Note executor contains the history of both experiments
    if executor.can_batch:
        assert executor.calls_to_executor == 2
    else:
        assert executor.calls_to_executor == 3
    assert len(executor.executed_circuits) == 3

    # For deterministic DDD sequences num_trials is irrelevant
    assert np.isclose(mitigated_1, mitigated_2)


def test_execute_with_ddd_with_full_output():
    """Tests the option full_output of execute_with_ddd."""
    executor = Executor(noiseless_serial_executor)

    ddd_value, ddd_data = execute_with_ddd(
        circuit_cirq_a,
        executor,
        rule=xx,
        num_trials=2,
        full_output=True,
    )
    assert len(executor.executed_circuits) == 2
    assert len(ddd_data["circuits_with_ddd"]) == 2
    assert len(ddd_data["ddd_trials"]) == 2
    assert ddd_data["ddd_value"] == ddd_value
    # For a deterministic rule
    assert ddd_data["ddd_trials"][0] == ddd_data["ddd_trials"][1]
