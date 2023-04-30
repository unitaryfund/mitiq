# Copyright (C) Unitary Fund
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, List, Optional, Tuple, Union, cast

import random
import cirq
import numpy as np

from mitiq import QPROGRAM, Executor, Observable, QuantumResult
from mitiq.interface import atomic_converter

# Paulis = Union[cirq.IdentityGate, cirq.XPowGate, cirq.YPowGate, cirq.ZPowGate]


def _generate_lookup_table(
    gate: str,
) -> List[Tuple[cirq.Gate, cirq.Gate, cirq.Gate, cirq.Gate]]:
    if gate not in ["CNOT", "CZ"]:
        raise ValueError("Invalid two-qubit gate. Supported gates: CNOT, CZ")

    c = cirq.X if gate == "CNOT" else cirq.Z
    lookup_table = [
        (cirq.I, cirq.I, cirq.I, cirq.I),
        (cirq.X, cirq.I, cirq.I, cirq.X),
        (cirq.Y, cirq.I, cirq.I, cirq.Y),
        (cirq.Z, cirq.I, cirq.I, cirq.Z),
        (cirq.I, c, c, cirq.I),
        (cirq.X, c, c, cirq.X),
        (cirq.Y, c, c, cirq.Y),
        (cirq.Z, c, c, cirq.Z),
        (cirq.I, cirq.Y, cirq.Y, cirq.I),
        (cirq.X, cirq.Y, cirq.Y, cirq.X),
        (cirq.Y, cirq.Y, cirq.Y, cirq.Y),
        (cirq.Z, cirq.Y, cirq.Y, cirq.Z),
        (cirq.I, cirq.Z, cirq.Z, cirq.I),
        (cirq.X, cirq.Z, cirq.Z, cirq.X),
        (cirq.Y, cirq.Z, cirq.Z, cirq.Y),
        (cirq.Z, cirq.Z, cirq.Z, cirq.Z),
    ]

    return lookup_table


lookup_table_CNOT = _generate_lookup_table("CNOT")
lookup_table_CZ = _generate_lookup_table("CZ")


def execute_with_pauli_twirling(
    circuit: QPROGRAM,
    executor: Union[Executor, Callable[[QPROGRAM], QuantumResult]],
    observable: Optional[Observable] = None,
    *,
    num_circuits: int = 10,
) -> float:
    executor = (
        executor if isinstance(executor, Executor) else Executor(executor)
    )
    twirled_circuits = pauli_twirled_circuits(circuit, num_circuits)
    expvals = executor.evaluate(twirled_circuits, observable)
    return cast(float, np.average(expvals))


def pauli_twirled_circuits(
    circuit: QPROGRAM, num_circuits: int
) -> List[QPROGRAM]:
    return [add_paulis(circuit) for _ in range(num_circuits)]


def sample_paulis(
    two_qubit_gate: cirq.Gate,
) -> Tuple[cirq.Gate, cirq.Gate, cirq.Gate, cirq.Gate]:
    if two_qubit_gate not in [cirq.CNOT, cirq.CZ]:
        raise ValueError("Invalid two-qubit gate. Supported gates: CNOT, CZ")

    lookup_table = (
        lookup_table_CNOT if two_qubit_gate == cirq.CNOT else lookup_table_CZ
    )
    P1, P2, R1, R2 = random.choice(lookup_table)
    return P1, P2, R1, R2


@atomic_converter
def add_paulis(circuit: cirq.Circuit) -> cirq.Circuit:
    return circuit.map_operations(twirl_two_qubit_gate)


def twirl_two_qubit_gate(op: cirq.Operation) -> cirq.OP_TREE:
    if op.gate not in [cirq.CNOT, cirq.CZ]:
        return op

    P, Q, R, S = sample_paulis(op.gate)
    control_qubit, target_qubit = op.qubits
    return [
        P.on(control_qubit),
        Q.on(target_qubit),
        op,
        R.on(control_qubit),
        S.on(target_qubit),
    ]
