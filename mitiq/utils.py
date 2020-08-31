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

"""Utility functions."""
from copy import deepcopy
from typing import cast

import numpy as np

from cirq import Circuit, CircuitDag, EigenGate, Gate, GateOperation, Moment
from cirq.ops.measurement_gate import MeasurementGate


def _simplify_gate_exponent(gate: EigenGate) -> EigenGate:
    """Returns the input gate with a simplified exponent if possible,
    otherwise the input gate is returned without any change.

    The input gate is not mutated.

    Args:
        gate: The input gate to simplify.

    Returns: The simplified gate.
    """
    # Try to simplify the gate exponent to 1
    if hasattr(gate, "_with_exponent") and gate == gate._with_exponent(1):
        return gate._with_exponent(1)
    return gate


def _simplify_circuit_exponents(circuit: Circuit) -> None:
    """Simplifies the gate exponents of the input circuit if possible,
    mutating the input circuit.

    Args:
        circuit: The circuit to simplify.
    """
    # Iterate over moments
    for moment_idx, moment in enumerate(circuit):
        simplified_operations = []
        # Iterate over operations in moment
        for op in moment:
            assert isinstance(op, GateOperation)
            if isinstance(op.gate, EigenGate):
                simplified_gate: Gate = _simplify_gate_exponent(op.gate)
            else:
                simplified_gate = op.gate
            simplified_operation = op.with_gate(simplified_gate)
            simplified_operations.append(simplified_operation)
        # Mutate the input circuit
        circuit[moment_idx] = Moment(simplified_operations)


def _equal(
    circuit_one: Circuit,
    circuit_two: Circuit,
    require_qubit_equality: bool = False,
    require_measurement_equality: bool = False,
) -> bool:
    """Returns True if the circuits are equal, else False.

    Args:
        circuit_one: Input circuit to compare to circuit_two.
        circuit_two: Input circuit to compare to circuit_one.
        require_qubit_equality: Requires that the qubits be equal
            in the two circuits.
        require_measurement_equality: Requires that measurements are equal on
            the two circuits, meaning that measurement keys are equal.

    Note:
        If set(circuit_one.all_qubits()) = {LineQubit(0)},
        then set(circuit_two_all_qubits()) must be {LineQubit(0)},
        else the two are not equal.
        If True, the qubits of both circuits must have a well-defined ordering.
    """
    if circuit_one is circuit_two:
        return True

    circuit_one = deepcopy(circuit_one)
    circuit_two = deepcopy(circuit_two)

    if not require_qubit_equality:
        # Transform the qubits of circuit one to those of circuit two
        qubit_map = dict(
            zip(
                sorted(circuit_one.all_qubits()),
                sorted(circuit_two.all_qubits()),
            )
        )
        circuit_one = circuit_one.transform_qubits(lambda q: qubit_map[q])

    if not require_measurement_equality:
        for circ in (circuit_one, circuit_two):
            measurements = [
                (moment, op)
                for moment, op, _ in circ.findall_operations_with_gate_type(
                    MeasurementGate
                )
            ]
            circ.batch_remove(measurements)

            for i in range(len(measurements)):
                gate = cast(MeasurementGate, measurements[i][1].gate)
                gate.key = ""

            circ.batch_insert(measurements)

    return CircuitDag.from_circuit(circuit_one) == CircuitDag.from_circuit(
        circuit_two
    )


def _are_close_dict(dict_a: dict, dict_b: dict) -> bool:
    """Returns True if the two dictionaries have equal keys and
    their corresponding values are "sufficiently" close."""
    keys_a = dict_a.keys()
    keys_b = dict_b.keys()
    if set(keys_a) != set(keys_b):
        return False
    for ka, va in dict_a.items():
        if not np.isclose(dict_b[ka], va):
            return False
    return True
