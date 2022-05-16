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
import copy
import numpy as np

from cirq.experiments.qubit_characterizations import (
    _single_qubit_cliffords,
    _random_single_q_clifford,
    _random_two_q_clifford,
    _gate_seq_to_mats,
    _find_inv_matrix,
    _two_qubit_clifford_matrices,
)
from typing import Sequence
from cirq import LineQubit, ops
from mitiq import QPROGRAM
from mitiq.interface import convert_from_mitiq


def generate_random_clifford_t_circuits(
    n_qubits: int,
    num_T: int,
    num_cliffords: int,
    trials: int,
    return_type: Optional[str] = None,
) -> List[QPROGRAM]:
    """Returns a list of random clifford + t circuits

    Args:
        num_T: The number of t gates in the circuit
        n_qubits: The number of qubits. Can be either 1 or 2.
        num_cliffords: The number of Clifford group elements in the
            random circuits. This is proportional to the depth per circuit.
        trials: The number of random circuits generated. The generation scheme is ran for (trials) times and a circuit generated each time.
        return_type: String which specifies the type of the
            returned circuits. See the keys of
            ``mitiq.SUPPORTED_PROGRAM_TYPES`` for options. If ``None``, the
            returned circuits have type ``cirq.Circuit``.

    Returns:
        A list of clifford + t circuits.
    """
    if n_qubits not in [1]:
        raise ValueError(
            "Only generates RB circuits on one"
            f"qubits not {n_qubits}."
        )
    qubits = LineQubit.range(n_qubits)
    cliffords = _single_qubit_cliffords()
    if n_qubits == 1:
        c1 = cliffords.c1_in_xy
        c10 = copy.deepcopy(c1)
        [gates.append(ops.Z**0.25) for gates in c10] #extended by T gates
        c1.extend(c10)

        cfd_mat_1q = cast(
            np.ndarray, [_gate_seq_to_mats(gates) for gates in c1]
        )
        circuits = [
            _random_single_q_clifford_T(qubits[0], num_cliffords, c1, cfd_mat_1q)
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

#function to generate single qubit cliffords
def _random_single_q_clifford_T(
    qubit: 'cirq.Qid',
    num_single_qubit_cliffords: int,
    cfds: Sequence[Sequence['cirq.Gate']],
    cfd_matrices: np.ndarray,
) -> 'cirq.Circuit':
    clifford_group_size = 24 #number of elements in clifford group
    gate_ids = list(np.random.choice(clifford_group_size, num_single_qubit_cliffords))
    gate_sequence = [[gate,ops.Z**0.25] for gate_id in gate_ids for gate in cfds[gate_id]]
    gate_sequence = [gate for list in gate_sequence for gate in list]
    idx = _find_inv_matrix(_gate_seq_to_mats(gate_sequence), cfd_matrices)
    gate_sequence.extend(cfds[idx])
    circuit = circuits.Circuit(gate(qubit) for gate in gate_sequence)
    return circuit