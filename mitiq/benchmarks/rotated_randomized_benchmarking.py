# Copyright (C) Unitary Fund
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

"""Functions for generating rotated randomized benchmarking circuits."""
from typing import List, Optional

import cirq
import numpy as np
from cirq.experiments.qubit_characterizations import (
    _find_inv_matrix,
    _gate_seq_to_mats,
    _single_qubit_cliffords,
    _two_qubit_clifford,
    _two_qubit_clifford_matrices,
)

from mitiq import QPROGRAM
from mitiq.interface import convert_from_mitiq


def generate_rotated_rb_circuits(
    n_qubits: int,
    num_cliffords: int,
    theta: Optional[float] = None,
    trials: int = 1,
    return_type: Optional[str] = None,
    seed: Optional[int] = None,
) -> List[QPROGRAM]:
    r"""
    Generates a list of "rotated" randomized benchmarking circuits.
    This benchmarking method enables testing QEM techniques in more general
    scenarios, closer to real-world applications in which expectation values
    can take arbitrary values.

    Rotated randomized bencmarking circuits are randomized benchmarking
    circuits with an :math:`R_z(\theta)` rotation inserted in the middle, such
    that:

    .. math::
        C(\theta) =    G_n \dots G_{n/2 +1} R_z(\theta)G_{n/2} \dots G_2 G_1

    where :math:`G_j` are Clifford elements or Clifford gates.

    For most values of the seed, the probability of the zero state is a
    sinusoidal function of :math:`\theta`. For some values of the seed
    the probability of the zero state is 1 for all :math:`\theta`.

    Since (up to factors of 2) we have
    :math:`R_z(\theta) =cos(\theta) I +  i \ sin(\theta) Z`, the rotated
    Clifford circuit :math:`C(\theta)` can be written as a linear combination
    of just two Clifford circuits, and therefore it is still easy to
    classically simulate.

    Args:
        n_qubits: The number of qubits. Can be either 1 or 2.
        num_cliffords: The number of Clifford group elements in the
            random circuits. This is proportional to the depth per circuit.
        theta: The rotation angle about the :math:`Z` axis.
        trials: The number of random circuits to return.
        return_type: String which specifies the type of the
            returned circuits. See the keys of
            ``mitiq.SUPPORTED_PROGRAM_TYPES`` for options. If ``None``, the
            returned circuits have type ``cirq.Circuit``.
        seed: A seed for generating radomzed benchmarking circuits.


    Returns:
        A list of rotated randomized benchmarking circuits.
    """

    if theta is None:
        theta = 2 * np.pi * np.random.rand()

    # return_type = "cirq" if not return_type else return_type
    # return [convert_from_mitiq(circuit, return_type) for circuit in circuits]
    if n_qubits not in (1, 2):
        raise ValueError(
            "Only generates RB circuits on one or two "
            f"qubits not {n_qubits}."
        )
    qubits = cirq.LineQubit.range(n_qubits)
    cliffords = _single_qubit_cliffords()
    rng = np.random.RandomState(seed)
    if n_qubits == 1:
        c1 = cliffords.c1_in_xy
        cfd_mat_1q = np.array(
            [_gate_seq_to_mats(gates) for gates in c1], dtype=np.complex64
        )
        circuits = []
        clifford_group_size = 24
        for _ in range(trials):
            gate_ids = list(rng.choice(clifford_group_size, num_cliffords))
            gate_sequence = [
                gate for gate_id in gate_ids for gate in c1[gate_id]
            ]
            gate_sequence.append(cirq.Rz(rads=theta).on(qubits[0]))
            idx = _find_inv_matrix(
                _gate_seq_to_mats(gate_sequence), cfd_mat_1q
            )
            gate_sequence.extend(c1[idx])
            circuits.append(
                cirq.Circuit(gate(qubits[0]) for gate in gate_sequence)
            )

    else:
        clifford_group_size = 11520
        cfd_matrices = _two_qubit_clifford_matrices(
            qubits[0],
            qubits[1],
            cliffords,
        )
        circuits = []
        for _ in range(trials):
            idx_list = list(rng.choice(clifford_group_size, num_cliffords))
            circuit = cirq.Circuit()
            for idx in idx_list:
                circuit.append(
                    _two_qubit_clifford(qubits[0], qubits[1], idx, cliffords)
                )
            circuit.append(cirq.Rz(rads=theta).on(qubits[0]))
            inv_idx = _find_inv_matrix(
                cirq.protocols.unitary(circuit), cfd_matrices
            )
            circuit.append(
                _two_qubit_clifford(qubits[0], qubits[1], inv_idx, cliffords)
            )

            circuits.append(circuit)

    return_type = "cirq" if not return_type else return_type
    return [convert_from_mitiq(circuit, return_type) for circuit in circuits]
