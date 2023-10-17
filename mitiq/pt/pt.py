# Copyright (C) Unitary Fund
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

import random
from typing import List

import cirq

from mitiq import QPROGRAM
from mitiq.interface import accept_qprogram_and_validate

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


def pauli_twirl_circuit(
    circuit: QPROGRAM,
    num_circuits: int = 10,
) -> List[QPROGRAM]:
    r"""Return the Pauli twirled versions of the input circuit.

    Only the $\mathrm{CZ}$ and $\mathrm{CNOT}$ gates in an
    input circuit are Pauli twirled as specified in
    :cite:`Saki_2023_arxiv`.

    Args:
        circuit: The input circuit to execute with twirling.
        num_circuits: Number of circuits to be twirled, and averaged.

    Returns:
        The expectation value estimated with Pauli twirling.
    """
    CNOT_twirled_circuits = twirl_CNOT_gates(circuit, num_circuits)
    twirled_circuits = [
        twirl_CZ_gates(c, num_circuits=1)[0] for c in CNOT_twirled_circuits
    ]

    return twirled_circuits


def twirl_CNOT_gates(circuit: QPROGRAM, num_circuits: int) -> List[QPROGRAM]:
    """Generate a list of circuits using Pauli twirling on CNOT gates.

    Args:
        circuit: The circuit to generate twirled versions of
        num_circuits: The number of sampled circuits to return
    """
    return [_twirl_CNOT_qprogram(circuit) for _ in range(num_circuits)]


@accept_qprogram_and_validate
def _twirl_CNOT_qprogram(circuit: cirq.Circuit) -> cirq.Circuit:
    return circuit.map_operations(_twirl_single_CNOT_gate)


def twirl_CZ_gates(circuit: QPROGRAM, num_circuits: int) -> List[QPROGRAM]:
    """Generate a list of circuits using Pauli twirling on CZ gates.

    Args:
        circuit: The circuit to generate twirled versions of
        num_circuits: The number of sampled circuits to return
    """
    return [_twirl_CZ_qprogram(circuit) for _ in range(num_circuits)]


@accept_qprogram_and_validate
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
