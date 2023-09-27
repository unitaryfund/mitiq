# Copyright (C) Unitary Fund
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

"""Functions for generating rotated randomized benchmarking circuits."""
from typing import List, Optional

import cirq
import numpy as np

from mitiq import QPROGRAM
from mitiq.benchmarks import generate_rb_circuits
from mitiq.interface import convert_from_mitiq, convert_to_mitiq


def generate_rotated_rb_circuits(
    n_qubits: int,
    num_cliffords: int,
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
    circuits in which an $R_z(\theta)$ rotation is inserted in the middle, such
    that:

    $$ C(\theta) =    G_n \dots G_{n/2 +1} R_z(\theta)G_{n/2} \dots G_2 G_1 $$

    where $G_j$ are Clifford elements or Clifford gates.

    The circuits generate expectation values which are sinusoidal functions of
    $\theta$, which in the ideal (noiseless) case vary in a continuous interval
    of $ E_{\rm ideal} \in  [-1, 1] $.

    Since (up to factors of 2) we have
    $R_z(\theta) =cos(\theta) I +  i \ sin(\theta) Z$, the rotated Clifford
    circuit $C(\theta)$ can be written as a linear combination of just two
    Clifford circuits and, therefore, it is still easy to classically simulate.

    Args:
        n_qubits: The number of qubits. Can be either 1 or 2.
        num_cliffords: The number of Clifford group elements in the
            random circuits. This is proportional to the depth per circuit.
        trials: The number of random circuits at each num_cfd.
        return_type: String which specifies the type of the
            returned circuits. See the keys of
            ``mitiq.SUPPORTED_PROGRAM_TYPES`` for options. If ``None``, the
            returned circuits have type ``cirq.Circuit``.
        seed: An optional seed for reproducibility of $\theta$ in
            $R_z(\theta)$.

    Returns:
        A list of rotated randomized benchmarking circuits.
    """

    circuits = generate_rb_circuits(n_qubits, num_cliffords, 2 * trials)
    rotated_circuits = []

    rng = np.random.default_rng(seed=seed)

    for circ in circuits:
        rotated_circ, _ = convert_to_mitiq(circ)
        qubits = rotated_circ.all_qubits()
        rads = rng.random() * np.pi
        rotated_circ.insert(
            len(circ) // 2, cirq.Rz(rads=rads).on_each(*qubits)
        )
        rotated_circuits.append(rotated_circ)

    return_type = "cirq" if not return_type else return_type
    return [
        convert_from_mitiq(circuit, return_type)
        for circuit in rotated_circuits
    ]