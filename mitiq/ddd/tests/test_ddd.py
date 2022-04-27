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

from mitiq import QPROGRAM, SUPPORTED_PROGRAM_TYPES
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

qreg_cirq = cirq.GridQubit.rect(10, 1)
circuit_cirq_b = cirq.Circuit(
    cirq.ops.H.on_each(*qreg_cirq), cirq.ops.H.on(qreg_cirq[1])
)


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
    # TODO fix Braket conversions of Pauli gates.
    if circuit_type == "braket":
        return

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
    # TODO fix Braket conversions of Pauli gates.
    if circuit_type == "braket":
        return

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
    # TODO fix Braket conversions of Pauli gates.
    if circuit_type == "braket":
        return

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
    assert not np.isclose(error_mitigated, error_unmitigated)
