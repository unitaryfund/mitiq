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

from mitiq import QPROGRAM, SUPPORTED_PROGRAM_TYPES, PauliString, Observable
from mitiq.interface import convert_to_mitiq, convert_from_mitiq

from mitiq.ddd.rules import xx, yy, xyxy
from mitiq.ddd import execute_with_ddd

from mitiq.pec.tests.test_pec import serial_executor, batched_executor, noiseless_serial_executor

circuit_cirq_a = cirq.Circuit(
    cirq.SWAP(q, q + 1) for q in cirq.LineQubit.range(7)
)

qreg_cirq = cirq.GridQubit.rect(10, 1)
circuit_cirq_b = cirq.Circuit(
    cirq.ops.H.on_each(*qreg_cirq), cirq.ops.H.on(qreg_cirq[1])
)


@mark.parametrize("circuit_type", SUPPORTED_PROGRAM_TYPES.keys())
@mark.parametrize("circuit", [circuit_cirq_a, circuit_cirq_b])
@mark.parametrize("rule", [xx, yy, xyxy])
def test_execute_with_ddd_without_noise(
    circuit_type, circuit, rule
):
    """Tests that execute_with_pec preserves expected results
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
@mark.parametrize("circuit", [circuit_cirq_a])
@mark.parametrize("executor", [serial_executor, batched_executor])
@mark.parametrize("rule", [xx, yy, xyxy])
def test_execute_with_ddd_mitigates_noise(
    circuit_type, circuit, executor, rule
):
    """Tests that execute_with_pec mitigates the error of a noisy
    expectation value.
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
    # TODO: restore tests when id_insertion is implemented.
    # assert error_mitigated < error_unmitigated
