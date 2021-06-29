# Copyright (C) 2021 Unitary Fund
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
"""Cirq utility functions."""

from typing import Union

import numpy as np
import cirq


# Note: This is the same as cirq.PauliSumLike but without a typing.ForwardRef
# which causes Sphinx errors.
PauliSumLike = Union[
    int,
    float,
    complex,
    cirq.PauliString,
    cirq.PauliSum,
    cirq.SingleQubitPauliStringGateOperation,
]


def execute(circuit: cirq.Circuit, obs: np.ndarray) -> float:
    """Simulates noiseless wavefunction evolution and returns the
    expectation value of some observable.

    Args:
        circuit: The input Cirq circuit.
        obs: The observable to measure as a NumPy array.

    Returns:
        The expectation value of obs as a float.
    """
    final_wvf = circuit.final_state_vector()
    return np.real(final_wvf.conj().T @ obs @ final_wvf)


def execute_with_shots(
    circuit: cirq.Circuit, obs: PauliSumLike, shots: int
) -> Union[float, complex]:
    """Simulates noiseless wavefunction evolution and returns the
    expectation value of a PauliString observable.

    Args:
        circuit: The input Cirq circuit.
        obs: The observable to measure as a cirq.PauliString.
        shots: The number of measurements.

    Returns:
        The expectation value of obs as a float.
    """

    # Do the sampling
    psum = cirq.PauliSumCollector(circuit, obs, samples_per_term=shots)
    psum.collect(sampler=cirq.Simulator())

    # Return the expectation value
    return psum.estimated_energy()


def execute_with_depolarizing_noise(
    circuit: cirq.Circuit, obs: np.ndarray, noise: float
) -> float:
    """Simulates a circuit with depolarizing noise at level noise.

    Args:
        circuit: The input Cirq circuit.
        obs: The observable to measure as a NumPy array.
        noise: The depolarizing noise as a float, i.e. 0.001 is 0.1% noise.

    Returns:
        The expectation value of obs as a float.
    """
    circuit = circuit.with_noise(cirq.depolarize(p=noise))  # type: ignore
    simulator = cirq.DensityMatrixSimulator()
    rho = simulator.simulate(circuit).final_density_matrix
    expectation = np.real(np.trace(rho @ obs))
    return expectation


def execute_with_shots_and_depolarizing_noise(
    circuit: cirq.Circuit, obs: PauliSumLike, noise: float, shots: int
) -> Union[float, complex]:
    """Simulates a circuit with depolarizing noise at level noise.

    Args:
        circuit: The input Cirq circuit.
        obs: The observable to measure as a NumPy array.
        noise: The depolarizing noise strength as a float (0.001 is 0.1%)
        shots: The number of measurements.

    Returns:
        The expectation value of obs as a float.
    """
    # Add noise
    noisy = circuit.with_noise(cirq.depolarize(p=noise))  # type: ignore

    # Do the sampling
    psum = cirq.PauliSumCollector(noisy, obs, samples_per_term=shots)
    psum.collect(sampler=cirq.DensityMatrixSimulator())

    # Return the expectation value
    return psum.estimated_energy()
