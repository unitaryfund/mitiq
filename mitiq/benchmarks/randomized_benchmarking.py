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
from typing import List

from cirq.experiments.qubit_characterizations import (
    _single_qubit_cliffords,
    _random_single_q_clifford,
    _random_two_q_clifford,
    _gate_seq_to_mats,
    _two_qubit_clifford_matrices,
)
from cirq import LineQubit, Circuit


def generate_rb_circuits(
    n_qubits: int, num_cliffords: int, trials: int = 1,
) -> List[Circuit]:
    """Returns a list of randomized benchmarking circuits, i.e. circuits that
    are equivalent to the identity.

    Args:
        n_qubits: The number of qubits. Can be either 1 or 2
        num_cliffords: The number of Clifford group elements in the
            random circuits. This is proportional to the depth per circuit.
        trials: The number of random circuits at each num_cfd.

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
        cfd_mat_1q = [_gate_seq_to_mats(gates) for gates in c1]

        return [
            _random_single_q_clifford(*qubits, num_cliffords, c1, cfd_mat_1q)
            for _ in range(trials)
        ]

    cfd_matrices = _two_qubit_clifford_matrices(
        *qubits, cliffords,  # type: ignore
    )
    return [
        _random_two_q_clifford(
            *qubits,  # type: ignore
            num_cliffords,
            cfd_matrices,
            cliffords,
        )
        for _ in range(trials)
    ]
