# Copyright (C) Unitary Fund
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

"""Functions for generating rotated randomized benchmarking circuits."""
from typing import Optional, Sequence, Tuple

import cirq
import numpy as np

# from cirq import S, H, CNOT, T, Circuit
from cirq.contrib.quantum_volume import compute_heavy_set
from cirq.value import big_endian_int_to_bits

from mitiq import QPROGRAM, Bitstring
from mitiq.interface import convert_from_mitiq


def generate_random_clifford_t_circuit(
    num_qubits: int,
    num_oneq_cliffords: int,
    num_twoq_cliffords: int,
    num_t_gates: int,
    return_type: Optional[str] = None,
    seed: Optional[int] = None,
) -> Tuple[QPROGRAM, Sequence[Bitstring]]:
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

    heavy_bitstrings = compute_heavy_bitstrings(circuit, num_qubits)

    return_type = "cirq" if not return_type else return_type
    return convert_from_mitiq(circuit, return_type), heavy_bitstrings


def compute_heavy_bitstrings(
    circuit: cirq.Circuit,
    num_qubits: int,
) -> Sequence[Bitstring]:
    """Classically compute the heavy bitstrings of the provided circuit.

    The heavy bitstrings are defined as the output bit-strings that have a
    greater than median probability of being generated.

    Args:
        circuit: The circuit to classically simulate.

    Returns:
        A list containing the heavy bitstrings.
    """
    heavy_vals = compute_heavy_set(circuit)
    # Convert base-10 ints to Bitstrings.
    heavy_bitstrings = [
        big_endian_int_to_bits(val, bit_count=num_qubits) for val in heavy_vals
    ]
    return heavy_bitstrings
