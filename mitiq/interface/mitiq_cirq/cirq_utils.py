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

from typing import Tuple

import numpy as np
import numpy.typing as npt
import cirq
from mitiq._typing import MeasurementResult


# Executors.
def sample_bitstrings(
    circuit: cirq.Circuit,
    noise_model: cirq.NOISE_MODEL_LIKE = cirq.amplitude_damp,  # type: ignore
    noise_level: Tuple[float] = (0.01,),
    sampler: cirq.Sampler = cirq.DensityMatrixSimulator(),
    shots: int = 8192,
) -> MeasurementResult:
    """Adds noise to the input circuit. The noise is added based on a
    particular noise model and some value for the error rate.

    Args:
        circuit: The input Cirq circuit.
        noise_model: Input Cirq noise model. Default is amplitude damping.
        noise_level: Noise rate as a tuple of floats.
        sampler: Cirq simulator from which the result will be sampled from.
        shots: Number of measurements.

    Returns:
        Sampled outcome from a measurement.
    """
    if sum(noise_level) > 0:
        circuit = circuit.with_noise(noise_model(*noise_level))  # type: ignore

    result = sampler.run(circuit, repetitions=shots)
    return MeasurementResult(
        result=np.column_stack(list(result.measurements.values())).tolist(),
        qubit_indices=tuple(
            # q[2:-1] is necessary to convert "q(number)" into "number"
            int(q[2:-1])
            for k in result.measurements.keys()
            for q in k.split(",")
        ),
    )


def compute_density_matrix(
    circuit: cirq.Circuit,
    noise_model: cirq.NOISE_MODEL_LIKE = cirq.amplitude_damp,  # type: ignore
    noise_level: Tuple[float] = (0.01,),
) -> npt.NDArray[np.complex64]:
    """Returns the density matrix of the quantum state after the
    (noisy) execution of the input circuit.

    Args:
        circuit: The input Cirq circuit.
        noise_model: Input Cirq noise model. Default is amplitude damping.
        noise_level: Noise rate as a tuple of floats.

    Returns:
        The final density matrix as a NumPy array.
    """
    if sum(noise_level) > 0:
        circuit = circuit.with_noise(noise_model(*noise_level))  # type: ignore

    return cirq.DensityMatrixSimulator().simulate(circuit).final_density_matrix


def execute_with_depolarizing_noise(
    circuit: cirq.Circuit, obs: npt.NDArray[np.complex64], noise: float
) -> float:
    """Simulates a circuit with depolarizing noise
    and returns the expectation value of the input observable.
    The expectation value is deterministically computed from
    the final density matrix and, therefore, shot noise is absent.

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
