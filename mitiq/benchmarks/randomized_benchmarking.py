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

"""Contains methods used for testing mitiq's performance on randomized
benchmarking circuits.
"""
from typing import List, Optional
import numpy as np

from cirq.experiments.qubit_characterizations import (
    _single_qubit_cliffords,
    _random_single_q_clifford,
    _random_two_q_clifford,
    _gate_seq_to_mats,
    _two_qubit_clifford_matrices,
)
from cirq import LineQubit, Circuit

CLIFFORDS = _single_qubit_cliffords()
C1 = CLIFFORDS.c1_in_xy
CFD_MAT_1Q = np.array([_gate_seq_to_mats(gates) for gates in C1])


def rb_circuits(
    n_qubits: int,
    num_cliffords: List[int],
    trials: int,
    qubit_labels: Optional[List[int]] = None,
) -> List[Circuit]:
    """Generates a set of randomized benchmarking circuits, i.e. circuits that
    are equivalent to the identity.

    Args:
        n_qubits: The number of qubits. Can be either 1 or 2
        num_cliffords: A list of numbers of Clifford group elements in the
            random circuits. This is proportional to the eventual depth per
            circuit.
        trials: The number of random circuits at each num_cfd.

    Returns:
        A list of randomized benchmarking circuits.
    """
    rb_circuits = []
    for num in num_cliffords:
        qid0 = qubit_labels[0] if qubit_labels else 0
        qubit1 = LineQubit(qid0)
        if n_qubits == 1:
            rb_circuits = [
                _random_single_q_clifford(qubit1, num, C1, CFD_MAT_1Q,)
                for _ in range(trials)
            ]
        elif n_qubits == 2:
            qid1 = qubit_labels[1] if qubit_labels else 1
            qubit2 = LineQubit(qid1)
            cfd_matrices = _two_qubit_clifford_matrices(
                qubit1, qubit2, CLIFFORDS,  # type: ignore
            )
            rb_circuits = [
                _random_two_q_clifford(
                    qubit1,  # type: ignore
                    qubit2,  # type: ignore
                    num,
                    cfd_matrices,
                    CLIFFORDS,
                )
                for _ in range(trials)
            ]
        else:
            raise ValueError(
                "Only generates RB circuits on one or two "
                f"qubits not {n_qubits}."
            )
    return rb_circuits
