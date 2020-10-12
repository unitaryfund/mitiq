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

"""Utility functions for benchmarking."""
from typing import cast

import numpy as np

from cirq import (
    Circuit,
    depolarize,
    DensityMatrixSimulator,
    DensityMatrixTrialResult,
)

SIMULATOR = DensityMatrixSimulator()


def noisy_simulation(circ: Circuit, noise: float, obs: np.ndarray) -> float:
    """Simulates a circuit with depolarizing noise at level NOISE.

    Args:
        circ: The quantum program as a cirq object.
        noise: The level of depolarizing noise.
        obs: The observable that the backend should measure.

    Returns:
        The observable's expectation value.
    """
    circuit = circ.with_noise(depolarize(p=noise))
    result = cast(DensityMatrixTrialResult, SIMULATOR.simulate(circuit))
    rho = result.final_density_matrix
    # measure the expectation by taking the trace of the density matrix
    expectation = np.real(np.trace(rho @ obs))
    return expectation
