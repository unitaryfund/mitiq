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

"""pyQuil utility functions."""
import numpy as np
from typing import Callable

from pyquil import Program
from pyquil.api import QuantumComputer
from pyquil.gates import MEASURE, RESET

from mitiq.mitiq_pyquil.compiler import basic_compile


def generate_qcs_executor(
    qc: QuantumComputer,
    expectation_fn: Callable[[np.ndarray], float],
    shots: int = 1000,
    reset: bool = True,
    debug: bool = False,
) -> Callable[[Program], float]:
    """Generates an executor for QCS that ingests pyQuil programs.

    Args:
        qc: The QuantumComputer object to use as backend.
        expectation_fn: Takes in bitstring results and produces a float.
        shots: Number of shots to take.
        reset: Whether or not to enable active reset.
        debug: If true, print the program after compilation.

    Returns:
        A customized executor function.
    """

    def executor(program: Program) -> float:
        p = Program()

        # add reset
        if reset:
            p += RESET()

        # add main body program
        p += program.copy()

        # add memory declaration
        qubits = p.get_qubits()
        ro = p.declare("ro", "BIT", len(qubits))

        # add measurements
        for idx, q in enumerate(qubits):
            p += MEASURE(q, ro[idx])

        # add numshots
        p.wrap_in_numshots_loop(shots)

        # nativize the circuit
        p = basic_compile(p)

        # print out nativized program
        if debug:
            print(p)

        # compile the circuit
        b = qc.compiler.native_quil_to_executable(p)

        # run the circuit, collect bitstrings
        qc.reset()
        results = qc.run(b)

        # compute expectation value
        return expectation_fn(results)

    return executor


def ground_state_expectation(results: np.ndarray) -> float:
    """
    Example expectation_fn. Computes the ground state expectation, also
    called survival probability.

    Args:
        results: Array of bitstrings from running a quantum program.

    Returns:
        A single expectation value computed from the results.
    """
    num_shots = len(results)
    return (
        num_shots - np.count_nonzero(np.count_nonzero(results, axis=1))
    ) / num_shots
