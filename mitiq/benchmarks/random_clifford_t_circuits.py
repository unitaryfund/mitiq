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

"""Functions for generating randomized benchmarking circuits."""
from typing import List, Optional, cast

import numpy as np

from cirq.experiments.qubit_characterizations import (
    _single_qubit_cliffords,
    _random_single_q_clifford,
    _random_two_q_clifford,
    _gate_seq_to_mats,
    _two_qubit_clifford_matrices,
)
from cirq import LineQubit
from mitiq import QPROGRAM
from mitiq.interface import convert_from_mitiq


def generate_random_clifford_t_circuits(
    num_T: int,
    num_single_qubit_cliffords: int,
    num_double_qubit_cliffords: int,
    trials: int = 1,
    return_type: Optional[str] = None,
) -> List[QPROGRAM]:
    """Returns a list of random clifford + t circuits

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
    print("start testing:") #remove this line
    qubits = LineQubit.range(2)
    cliffords = _single_qubit_cliffords()
    print("cliffords", cliffords)

    c1 = cliffords.c1_in_xy
    cfd_mat_1q = cast(
        np.ndarray, [_gate_seq_to_mats(gates) for gates in c1]
    )
    print("matrixs", cfd_mat_1q)
    circuits = [
        _random_single_q_clifford(qubits[0], num_single_qubit_cliffords, c1, cfd_mat_1q)
        for _ in range(trials)
    ]
    cfd_matrices = _two_qubit_clifford_matrices(
        qubits[0],
        qubits[1],
        cliffords,
    )
    circuits = [
        _random_two_q_clifford(
            qubits[0],
            qubits[1],
            num_double_qubit_cliffords,
            cfd_matrices,
            cliffords,
        )
        for _ in range(trials)
    ]

    return_type = "cirq" if not return_type else return_type
    return [convert_from_mitiq(circuit, return_type) for circuit in circuits]
