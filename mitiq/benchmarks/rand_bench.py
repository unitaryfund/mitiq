# random_circ.py
"""
Contains methods used for testing mitiq's performance on randomized
benchmarking circuits.
"""
from typing import List, Tuple
import numpy as np

from cirq.experiments.qubit_characterizations import _single_qubit_cliffords, \
    _random_single_q_clifford, _random_two_q_clifford, _gate_seq_to_mats, \
    _two_qubit_clifford_matrices
from cirq import NamedQubit, Circuit

cliffords = _single_qubit_cliffords()
c1 = cliffords.c1_in_xy
cfd_mats = np.array([_gate_seq_to_mats(gates) for gates in c1])


def rb_circuits(n_qubits: int, num_cfds: List[int], trials: int) -> \
        List[Circuit]:
    rb_circuits = []
    for num in num_cfds:
        qubit1 = NamedQubit("0")
        if n_qubits == 1:
            rb_circuits = [
                _random_single_q_clifford(qubit1, num, c1, cfd_mats)
                for _ in range(trials)
            ]
        elif n_qubits == 2:
            qubit2 = NamedQubit("1")
            cfd_matrices = _two_qubit_clifford_matrices(qubit1, qubit2,
                                                        cliffords)
            rb_circuits = [
                _random_two_q_clifford(qubit1, qubit2, num, cfd_matrices,
                                       cliffords)
                for _ in range(trials)
            ]
        else:
            raise ValueError("Only generates RB circuits on one or two "
                             f"qubits not {n_qubits}.")
    return rb_circuits
