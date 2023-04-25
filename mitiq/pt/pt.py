# Copyright (C) Unitary Fund
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, List, Optional, Tuple, Union

import cirq
import numpy as np

from mitiq import QPROGRAM, Executor, Observable, QuantumResult
from mitiq.interface import atomic_converter


def execute_with_pauli_twirling(
    circuit: QPROGRAM,
    executor: Union[Executor, Callable[[QPROGRAM], QuantumResult]],
    observable: Optional[Observable] = None,
    *,
    num_circuits: int = 100,
) -> float:
    executor = (
        executor if isinstance(executor, Executor) else Executor(executor)
    )
    twirled_circuits = pauli_twirled_circuits(circuit, num_circuits)
    expvals = executor.evaluate(twirled_circuits, observable)
    return np.average(expvals)


def pauli_twirled_circuits(
    circuit: QPROGRAM, num_circuits: int
) -> List[QPROGRAM]:
    return [add_paulis(circuit) for _ in range(num_circuits)]


Paulis = Union[cirq.IdentityGate, cirq.XPowGate, cirq.YPowGate, cirq.ZPowGate]


def sample_paulis() -> Tuple[Paulis, Paulis, Paulis, Paulis]:
    # P, Q, R, S
    return cirq.ops.X, cirq.ops.I, cirq.ops.X, cirq.ops.X


@atomic_converter
def add_paulis(circuit: cirq.Circuit) -> cirq.Circuit:
    return circuit.map_operations(twirl_CNOT)


def twirl_CNOT(op: cirq.Operation) -> cirq.OP_TREE:
    if op.gate != cirq.CNOT:
        return op

    P, Q, R, S = sample_paulis()
    control_qubit, target_qubit = op.qubits
    return [
        P.on(control_qubit),
        Q.on(target_qubit),
        op,
        R.on(control_qubit),
        S.on(target_qubit),
    ]
