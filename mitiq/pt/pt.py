# Copyright (C) Unitary Fund
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, List, Optional, Union, cast

import random
import cirq
import numpy as np

from mitiq import QPROGRAM, Executor, Observable, QuantumResult
from mitiq.interface import atomic_converter

# P, Q, R, S from https://arxiv.org/pdf/2301.02690.pdf
CNOT_twirling_gates = [
    (cirq.I, cirq.I, cirq.I, cirq.I),
    (cirq.I, cirq.X, cirq.I, cirq.X),
    (cirq.I, cirq.Y, cirq.Z, cirq.Y),
    (cirq.I, cirq.Z, cirq.Z, cirq.Z),
    (cirq.Y, cirq.I, cirq.Y, cirq.X),
    (cirq.Y, cirq.X, cirq.Y, cirq.I),
    (cirq.Y, cirq.Y, cirq.X, cirq.Z),
    (cirq.Y, cirq.Z, cirq.X, cirq.Y),
    (cirq.X, cirq.I, cirq.X, cirq.X),
    (cirq.X, cirq.X, cirq.X, cirq.I),
    (cirq.X, cirq.Y, cirq.Y, cirq.Z),
    (cirq.X, cirq.Z, cirq.Y, cirq.Y),
    (cirq.Z, cirq.I, cirq.Z, cirq.I),
    (cirq.Z, cirq.X, cirq.Z, cirq.X),
    (cirq.Z, cirq.Y, cirq.I, cirq.Y),
    (cirq.Z, cirq.Z, cirq.I, cirq.Z),
]
# TODO: check the CZ gates. should be in PQRS order
CZ_twirling_gates = [
    (cirq.I, cirq.I, cirq.I, cirq.I),
    (cirq.X, cirq.I, cirq.I, cirq.X),
    (cirq.Y, cirq.I, cirq.I, cirq.Y),
    (cirq.Z, cirq.I, cirq.I, cirq.Z),
    (cirq.I, cirq.Z, cirq.Z, cirq.I),
    (cirq.X, cirq.Z, cirq.Z, cirq.X),
    (cirq.Y, cirq.Z, cirq.Z, cirq.Y),
    (cirq.Z, cirq.Z, cirq.Z, cirq.Z),
    (cirq.I, cirq.Y, cirq.Y, cirq.I),
    (cirq.X, cirq.Y, cirq.Y, cirq.X),
    (cirq.Y, cirq.Y, cirq.Y, cirq.Y),
    (cirq.Z, cirq.Y, cirq.Y, cirq.Z),
    (cirq.I, cirq.Z, cirq.Z, cirq.I),
    (cirq.X, cirq.Z, cirq.Z, cirq.X),
    (cirq.Y, cirq.Z, cirq.Z, cirq.Y),
    (cirq.Z, cirq.Z, cirq.Z, cirq.Z),
]


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
    twirled_circuits = twirl_CNOT_gates(circuit, num_circuits)
    expvals = executor.evaluate(twirled_circuits, observable)
    return cast(float, np.average(expvals))


def twirl_CNOT_gates(circuit: QPROGRAM, num_circuits: int) -> List[QPROGRAM]:
    twirl_qprogram = atomic_converter(
        lambda circuit: circuit.map_operations(_twirl_single_CNOT_gate)
    )
    return [twirl_qprogram(circuit) for _ in range(num_circuits)]


def twirl_CZ_gates(circuit: QPROGRAM, num_circuits: int) -> List[QPROGRAM]:
    twirl_qprogram = atomic_converter(
        lambda circuit: circuit.map_operations(_twirl_single_CZ_gate)
    )
    return [twirl_qprogram(circuit) for _ in range(num_circuits)]


def _twirl_single_CNOT_gate(op: cirq.Operation) -> cirq.OP_TREE:
    if op.gate != cirq.CNOT:
        return op

    P, Q, R, S = random.choice(CNOT_twirling_gates)
    control_qubit, target_qubit = op.qubits
    return [
        P.on(control_qubit),
        Q.on(target_qubit),
        op,
        R.on(control_qubit),
        S.on(target_qubit),
    ]


def _twirl_single_CZ_gate(op: cirq.Operation) -> cirq.OP_TREE:
    if op.gate != cirq.CZ:
        return op

    P, Q, R, S = random.choice(CZ_twirling_gates)
    control_qubit, target_qubit = op.qubits
    return [
        P.on(control_qubit),
        Q.on(target_qubit),
        op,
        R.on(control_qubit),
        S.on(target_qubit),
    ]
