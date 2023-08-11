# Copyright (C) Unitary Fund
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

"""Functions for generating randomized benchmarking circuits."""
from typing import List, Optional

import numpy as np
from cirq import LineQubit
from cirq.experiments.qubit_characterizations import (
    _gate_seq_to_mats,
    _random_single_q_clifford,
    _random_two_q_clifford,
    _single_qubit_cliffords,
    _two_qubit_clifford_matrices,
)

from mitiq import QPROGRAM
from mitiq.interface import convert_from_mitiq


def generate_rb_circuits(
    n_qubits: int,
    num_cliffords: int,
    trials: int = 1,
    return_type: Optional[str] = None,
) -> List[QPROGRAM]:
    """Returns a list of randomized benchmarking circuits, i.e. circuits that
    are equivalent to the identity.

    Args:
        n_qubits: The number of qubits. Can be either 1 or 2.
        num_cliffords: The number of Clifford group elements in the
            random circuits. This is proportional to the depth per circuit.
        trials: The number of random circuits at each num_cfd.
        return_type: String which specifies the type of the
            returned circuits. See the keys of
            ``mitiq.SUPPORTED_PROGRAM_TYPES`` for options. If ``None``, the
            returned circuits have type ``cirq.Circuit``.

    Returns:
        A list of randomized benchmarking circuits.
    """
    if n_qubits not in (1, 2):
        raise ValueError(
            "Only generates RB circuits on one or two "
            f"qubits not {n_qubits}."
        )
    qubits = LineQubit.range(n_qubits)
    cliffords = _single_qubit_cliffords()

    if n_qubits == 1:
        c1 = cliffords.c1_in_xy
        cfd_mat_1q = np.array(
            [_gate_seq_to_mats(gates) for gates in c1], dtype=np.complex64
        )

        circuits = [
            _random_single_q_clifford(qubits[0], num_cliffords, c1, cfd_mat_1q)
            for _ in range(trials)
        ]
    else:
        cfd_matrices = _two_qubit_clifford_matrices(
            qubits[0],
            qubits[1],
            cliffords,
        )
        circuits = [
            _random_two_q_clifford(
                qubits[0],
                qubits[1],
                num_cliffords,
                cfd_matrices,
                cliffords,
            )
            for _ in range(trials)
        ]

    return_type = "cirq" if not return_type else return_type
    return [convert_from_mitiq(circuit, return_type) for circuit in circuits]
