# Copyright (C) Unitary Fund
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, List, Optional, Union, cast

import random
import cirq
import numpy as np

from mitiq import QPROGRAM, Executor, Observable, QuantumResult
from mitiq.interface import noise_scaling_converter

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
CZ_twirling_gates = [
    (cirq.I, cirq.I, cirq.I, cirq.I),
    (cirq.I, cirq.X, cirq.Z, cirq.X),
    (cirq.I, cirq.Y, cirq.Z, cirq.Y),
    (cirq.I, cirq.Z, cirq.I, cirq.Z),
    (cirq.X, cirq.I, cirq.X, cirq.Z),
    (cirq.X, cirq.X, cirq.Y, cirq.Y),
    (cirq.X, cirq.Y, cirq.Y, cirq.X),
    (cirq.X, cirq.Z, cirq.X, cirq.I),
    (cirq.Y, cirq.I, cirq.Y, cirq.Z),
    (cirq.Y, cirq.X, cirq.X, cirq.Y),
    (cirq.Y, cirq.Y, cirq.X, cirq.X),
    (cirq.Y, cirq.Z, cirq.Y, cirq.I),
    (cirq.Z, cirq.I, cirq.Z, cirq.I),
    (cirq.Z, cirq.X, cirq.I, cirq.X),
    (cirq.Z, cirq.Y, cirq.I, cirq.Y),
    (cirq.Z, cirq.Z, cirq.Z, cirq.Z),
]


def execute_with_pauli_twirling(
    circuit: QPROGRAM,
    executor: Union[Executor, Callable[[QPROGRAM], QuantumResult]],
    observable: Optional[Observable] = None,
    *,
    num_circuits: int = 10,
) -> float:
    """Estimates the expectation value of the input circuit by averaging
    expectation values obtained from Pauli twirled circuits.

    Args:
        circuit: The input circuit to execute with twirling.
        executor: A Mitiq executor that executes a circuit and returns the
            unmitigated ``QuantumResult`` (e.g. an expectation value).
        observable: Observable to compute the expectation value of. If
            ``None``, the ``executor`` must return an expectation value
            (float). Otherwise, the ``QuantumResult`` returned by ``executor``
            is used to compute the expectation of the observable.
        num_circuits: Number of circuits to be twirled, and averaged.

    Returns:
        The expectation value estimated with Pauli twirling.
    """
    executor = (
        executor if isinstance(executor, Executor) else Executor(executor)
    )
    CNOT_twirled_circuits = twirl_CNOT_gates(circuit, num_circuits)
    twirled_circuits = [
        twirl_CZ_gates(c, num_circuits=1)[0] for c in CNOT_twirled_circuits
    ]
    expvals = executor.evaluate(twirled_circuits, observable)
    return cast(float, np.average(expvals))


def twirl_CNOT_gates(circuit: QPROGRAM, num_circuits: int) -> List[QPROGRAM]:
    """Generate a list of circuits using Pauli twirling on CNOT gates.

    Args:
        circuit: The circuit to generate twirled versions of
        num_circuits: The number of sampled circuits to return
    """
    return [_twirl_CNOT_qprogram(circuit) for _ in range(num_circuits)]


@noise_scaling_converter
def _twirl_CNOT_qprogram(circuit: cirq.Circuit) -> cirq.Circuit:
    return circuit.map_operations(_twirl_single_CNOT_gate)


def twirl_CZ_gates(circuit: QPROGRAM, num_circuits: int) -> List[QPROGRAM]:
    """Generate a list of circuits using Pauli twirling on CZ gates.

    Args:
        circuit: The circuit to generate twirled versions of
        num_circuits: The number of sampled circuits to return
    """
    return [_twirl_CZ_qprogram(circuit) for _ in range(num_circuits)]


@noise_scaling_converter
def _twirl_CZ_qprogram(circuit: cirq.Circuit) -> cirq.Circuit:
    return circuit.map_operations(_twirl_single_CZ_gate)


def _twirl_single_CNOT_gate(op: cirq.Operation) -> cirq.OP_TREE:
    """Function which converts a CNOT gate to a logical equivalent. That is,
    it converts CNOT operations into the following, leaving all other cirq
    operations alone.

        --P---⏺---R--
              |
        --Q---⊕---S--

    Where P, Q, R, and S are Pauli operators sampled so as to not effect
    change on the underlying unitary.
    """
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
    """Function which converts a CZ gate to a logical equivalent. That is, it
    converts CZ operations into the following, leaving all other cirq
    operations alone.

        --P---⏺---R--
              |
        --Q---⏺---S--

    Where P, Q, R, and S are Pauli operators sampled so as to not effect
    change on the underlying unitary.
    """
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
