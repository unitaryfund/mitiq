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

"""Utilities for generating benchmarking programs.
"""

from typing import List, Optional, Sequence

import numpy as np

from pyquil import Program
from pyquil.api import BenchmarkConnection
from pyquil.gates import CZ, RX, RZ
from pyquil.quilbase import Gate

NATIVE_1Q_GATES = [RX, RZ]
MAGIC_ANGLES = [-np.pi, -np.pi / 2, np.pi / 2, np.pi]


def one_qubit_gateset(qubit: int) -> List[Gate]:
    """
    Return the RX, RZ native gateset on one qubit.

    Args:
        qubit: The qubit to build a 1Q gateset with.

    Returns:
        A list of Gate objects representing an (RX, RZ) 1Q gateset.
    """
    return list(
        gate(angle, qubit)
        for gate in NATIVE_1Q_GATES
        for angle in MAGIC_ANGLES
    )


def two_qubit_gateset(q0: int, q1: int) -> List[Gate]:
    """
    Return the RX, RZ, CZ native gateset on two qubits.

    Args:
        q0: One of the two qubits to use for the gateset.
        q1: The other qubit to use for the gateset.

    Returns:
        A list of Gate objects representing an (RX, RZ, CZ) 2Q gateset.
    """
    return one_qubit_gateset(q0) + one_qubit_gateset(q1) + [CZ(q0, q1)]


def generate_rb_program(
    benchmarker: BenchmarkConnection,
    qubits: Sequence[int],
    depth: int,
    interleaved_gate: Optional[Program] = None,
    random_seed: Optional[int] = None,
) -> Program:
    """
    Generate a randomized benchmarking program.

    Args:
        benchmarker: Connection object to quilc for generating sequences.
        qubits: The qubits to generate and RB sequence for.
        depth: Total number of Cliffords in the sequence (with inverse).
        interleaved_gate: Gate to interleave into the sequence for IRB.
        random_seed: Random seed passed to the benchmarker.

    Returns:
        A pyQuil Program for a randomized benchmarking sequence.
    """
    if depth < 2:
        raise ValueError("Sequence depth must be at least 2 for RB sequences.")

    if len(qubits) == 1:
        gateset = one_qubit_gateset(qubits[0])
    elif len(qubits) == 2:
        gateset = two_qubit_gateset(*qubits)
    else:
        raise ValueError("We only support one- and two-qubit RB.")

    programs = benchmarker.generate_rb_sequence(
        depth=depth,
        gateset=gateset,
        interleaver=interleaved_gate,
        seed=random_seed,
    )
    return Program(programs)
