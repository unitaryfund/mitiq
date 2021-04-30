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


import numpy as np
import cirq


def execute(circ: cirq.Circuit, obs: np.ndarray) -> float:
    """Simulates noiseless wavefunction evolution and returns the
        expectation value of some observable.

        Args:
            circ: The input Cirq circuit.
            obs: The observable to measure as a NumPy array.

        Returns:
            The expectation value of obs as a float.
        """
    final_wvf = circ.final_state_vector()
    return np.real(final_wvf.conj().T @ obs @ final_wvf)


def execute_with_shots(
    circ: cirq.Circuit, obs: cirq.PauliString, shots: int
) -> float:
    """Simulates noiseless wavefunction evolution and returns the
        expectation value of a PauliString observable.

        Args:
            circ: The input Cirq circuit.
            obs: The observable to measure as a cirq.PauliString.
            shots: The number of measurements.

        Returns:
            The expectation value of obs as a float.
        """

    # Do the sampling
    psum = cirq.PauliSumCollector(circ, obs, samples_per_term=shots)
    psum.collect(sampler=cirq.Simulator())

    # Return the expectation value
    return psum.estimated_energy()


def execute_with_depolarizing_noise(
    circ: cirq.Circuit, obs: np.ndarray, noise: float
) -> float:
    """Simulates a circuit with depolarizing noise at level noise.

        Args:
            circ: The input Cirq circuit.
            obs: The observable to measure as a NumPy array.
            noise: The depolarizing noise as a float, i.e. 0.001 is 0.1% noise.

        Returns:
            The expectation value of obs as a float.
        """
    circuit = circ.with_noise(cirq.depolarize(p=noise))
    simulator = cirq.DensityMatrixSimulator()
    rho = simulator.simulate(circuit).final_density_matrix
    expectation = np.real(np.trace(rho @ obs))
    return expectation


def execute_with_shots_and_depolarizing_noise(
    circ: cirq.Circuit, obs: cirq.PauliString, noise: float, shots: int
) -> float:
    """Simulates a circuit with depolarizing noise at level noise.

        Args:
            circ: The input Cirq circuit.
            obs: The observable to measure as a NumPy array.
            noise: The depolarizing noise strength as a float (0.001 is 0.1%)
            shots: The number of measurements.

        Returns:
            The expectation value of obs as a float.
        """
    # Add noise
    noisy = circ.with_noise(cirq.depolarize(p=noise))

    # Do the sampling
    psum = cirq.PauliSumCollector(noisy, obs, samples_per_term=shots)
    psum.collect(sampler=cirq.DensityMatrixSimulator())

    # Return the expectation value
    return psum.estimated_energy()
