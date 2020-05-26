# random_circ.py
"""
Contains methods used for testing mitiq's performance on randomized
benchmarking circuits.
"""
from typing import List
import numpy as np

from cirq.experiments.qubit_characterizations import _single_qubit_cliffords, \
    _random_single_q_clifford, _random_two_q_clifford, _gate_seq_to_mats, \
    _two_qubit_clifford_matrices
from cirq import NamedQubit, Circuit

CLIFFORDS = _single_qubit_cliffords()
C1 = CLIFFORDS.c1_in_xy
CFD_MAT_1Q = np.array([_gate_seq_to_mats(gates) for gates in C1])


def rb_circuits(n_qubits: int, num_cfds: List[int], trials: int) -> \
        List[Circuit]:
    """Generates a set of randomized benchmarking circuits, i.e. circuits that
    are equivalent to the identity.

    Args:
        n_qubits: The number of qubits. Can be either 1 or 2
        num_cfds: A list of numbers of Clifford group elements in the random
                  circuits. This is proportional to the eventual depth
                  per circuit.
        trials: The number of random circuits at each num_cfd.

    Returns:
        A list of randomized benchmarking circuits
    """
    rb_circuits = []
    for num in num_cfds:
        qubit1 = NamedQubit("0")
        if n_qubits == 1:
            rb_circuits = [
                _random_single_q_clifford(qubit1, num, C1, CFD_MAT_1Q)
                for _ in range(trials)
            ]
        elif n_qubits == 2:
            qubit2 = NamedQubit("1")
            cfd_matrices = _two_qubit_clifford_matrices(qubit1, qubit2,
                                                        CLIFFORDS)
            rb_circuits = [
                _random_two_q_clifford(qubit1, qubit2, num, cfd_matrices,
                                       CLIFFORDS)
                for _ in range(trials)
            ]
        else:
            raise ValueError("Only generates RB circuits on one or two "
                             f"qubits not {n_qubits}.")
    return rb_circuits
