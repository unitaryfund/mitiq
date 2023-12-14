# Copyright (C) Unitary Fund
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

"""Functions for generating rotated randomized benchmarking circuits."""
from typing import Optional

import cirq
import numpy as np

from mitiq import QPROGRAM
from mitiq.interface import convert_from_mitiq


def generate_random_clifford_t_circuit(
    num_qubits: int,
    num_oneq_cliffords: int,
    num_twoq_cliffords: int,
    num_t_gates: int,
    return_type: Optional[str] = None,
    seed: Optional[int] = None,
) -> QPROGRAM:
    r"""Generate a random quantum circuit with the given number of qubits,
    number of one-qubit Cliffords, number of two-qubit Cliffords and number
    of T gates.

    Args:
        num_qubits: The number of qubits in the generated circuit.
        num_oneq_cliffords: Number of one-qubit Cliffords to be used.
        num_twoq_cliffords: Number of two-qubit Cliffords to be used.
        num_t_gates: Number of T gates to be used.
        seed: Seed for generating random circuit.
        return_type: String which specifies the type of the returned
            circuits. See the keys of ``mitiq.SUPPORTED_PROGRAM_TYPES``
            for options. If ``None``, the returned circuits have type
            ``cirq.Circuit``.

    Returns:
        A quantum circuit acting on ``num_qubits`` qubits.
        A list of the heavy bitstrings for the returned circuit.
    """

    if num_qubits <= 0:
        raise ValueError(
            "Cannot prepare a circuit with {} qubits.", num_qubits
        )
    elif num_qubits == 1 and num_twoq_cliffords > 0:
        raise ValueError(
            "Need more than 2 qubits for two-qubit Clifford gates."
        )

    rnd_state = np.random.RandomState(seed)

    oneq_cliffords = [cirq.S, cirq.H]
    twoq_cliffords = [cirq.CNOT, cirq.CZ]
    oneq_list = [
        rnd_state.choice(oneq_cliffords) for _ in range(num_oneq_cliffords)
    ]
    twoq_list = [
        rnd_state.choice(twoq_cliffords) for _ in range(num_twoq_cliffords)
    ]
    t_list = [cirq.T for _ in range(num_t_gates)]

    all_gates = oneq_list + twoq_list + t_list
    rnd_state.shuffle(all_gates)

    qubits = cirq.LineQubit.range(num_qubits)
    circuit = cirq.Circuit()

    for gate in all_gates:
        qubits_for_gate = rnd_state.choice(
            qubits, size=gate.num_qubits(), replace=False
        )
        operation = gate.on(*qubits_for_gate)
        circuit.append(operation)

    return_type = "cirq" if not return_type else return_type
    return convert_from_mitiq(circuit, return_type)
