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

from typing import Counter, Dict, List, Union

import numpy as np

MeasurementResult = Union[Dict[bin, int], Counter[bin]]


def calculate_observable(
    state_or_measurements: Union[MeasurementResult, np.ndarray], observable: np.ndarray
) -> float:
    """Returns (estimate of) ⟨𝛹| O |𝛹⟩ for diagonal observable O and quantum
     state |𝛹⟩.

    Args:
        state_or_measurements: Quantum state to calculate the expectation
            value of the observable in. Can be provided as a wavefunction
            (numpy array) or as a dictionary of counts from sampling the
            wavefunction in the computational basis.
        observable: Observable as a diagonal matrix (one-dimensional numpy
            array).
    """
    nqubits = int(np.log2(len(observable)))

    if isinstance(state_or_measurements, np.ndarray):
        observable_values = [
            observable[i] * abs(np.conjugate(state_or_measurements[i]) * state_or_measurements[i])
            for i in range(2 ** nqubits)
        ]
    elif isinstance(state_or_measurements, dict):
        # order the counts and add zeros:
        state_or_measurements = measurements_to_probabilities(state_or_measurements, nqubits)
        values = list(state_or_measurements.values())
        observable_values = [
            (observable[i] * values[i]) for i in range(2 ** nqubits)
        ]
    else:
        raise ValueError(
            f"Provided state has type {type(state_or_measurements)} but must be a numpy "
            f"array or dictionary of counts."
        )

    return sum(np.real(observable_values))


def construct_training_data_floats(
    training_data: List[MeasurementResult], observable: np.ndarray,
) -> (np.ndarray, np.ndarray):
    """Function to calculate training data now as two arrays of floats to be
    used in the regression (raw_training_data, simulated_training_data).
    Args:
        training_data: List of dictionary of counts for all training circuits
                       and all noise levels. In the form:

        ([List[dict] (simulated data), [List[List[dict]] (real circuit data)])

        observable: Option to be passed to use defined observable function that
                    defines how to calculate the value of an observable from
                    the counts.
    Returns: Tuple of np.ndarray of dimensions
            (num_training_circuits x noise_levels) and (num_training_circuits).
    """
    training_circuits_raw_data = training_data[1]
    training_circuits_simulated_data = training_data[0]
    noise_levels = len(training_circuits_raw_data)
    number_of_training_circuits = len(training_circuits_simulated_data)

    # first need to sort the training data, then will do a regression.
    X_data = np.zeros((number_of_training_circuits, noise_levels))
    Y_data = np.zeros(number_of_training_circuits)
    for i, training_circuit_raw_data_one_noise_level in enumerate(
        training_circuits_raw_data
    ):
        for j, training_circuit_dict in enumerate(
            training_circuit_raw_data_one_noise_level
        ):
            training_obs_raw = calculate_observable(
                training_circuit_dict, observable
            )
            X_data[j, i] = training_obs_raw
            if i == 0:
                Y_data[j] = calculate_observable(
                    training_circuits_simulated_data[j], observable
                )
    return (X_data, Y_data)


def construct_circuit_data_floats(
    circuit_data: List[MeasurementResult], observable: np.ndarray
) -> np.ndarray:
    """Returns circuit of interest now as two arrays of floats.
    Args:
        circuit_data: List of dictionary of counts for circuit of interest
                      and all noise levels.
        observable: Option to be passed to use defined observable function that
                    defines how to calculate the value of an observable from
                    the counts.
    Returns: Array of floats for observable calculated from input data.
    """
    return np.array([
        calculate_observable(state_or_measurements=measurements, observable=observable)
        for measurements in circuit_data
    ])


def measurements_to_probabilities(
    counts: MeasurementResult, nqubits: int,
) -> MeasurementResult:
    """Normalizes the counts and inserts 0 where counts are missing."""
    total_counts = sum(counts.values())
    state = {bin(j): 0.0 for j in range(2 ** nqubits)}
    for key, value in counts.items():
        state[key] = value / total_counts
    return state
