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

""" Functions for using near-Clifford training data for error mitigation"""
from typing import List, Union, Callable, Dict

import numpy as np

from cirq.circuits import Circuit


def scale_noise_in_circuits(
    circuits: List[Circuit],
    scale_noise: Callable[[Circuit, float], Circuit],
    scale_factor: Union[float, List],
) -> List[List[Circuit]]:
    """Function to scale the noise in a list of circuits.
    Args:
        circuits: list of training circuits.
        scale_factor: factor by which to scale the circuits.
        scale_noise: method to use to fold the circuits.
    Returns: List os lists containing the original circuits followed by the
             folded circuits: [[circuits], [folded_circuits], ...].
    """
    all_folded_circuits = [circuits]
    if isinstance(scale_factor, float) or isinstance(scale_factor, int):
        scale_factor = [scale_factor]
    for scale_factor_ in scale_factor:
        folded_circuits = []
        for circuit in circuits:
            folded_circuits.append(scale_noise(circuit, scale_factor_))
        all_folded_circuits.append(folded_circuits)
    return all_folded_circuits


def calculate_observable(
    state: Union[dict, np.ndarray], observable: np.ndarray
) -> float:
    """ Function to take a users definition of an observable and calculate its
    value from the counts passed into the function (counts could be simulated
    or from raw data).
    Args:
        state: the state represented in raw or simulated counts with which to
               extract the observable or as a statevector.
        observable: array of diagonal elements of observable to be measured,
                    which is a diagonal matrix.
    Returns: observable calculated from dictionaries or statevectors.
    """
    nqubits = int(np.log2(len(observable)))
    if isinstance(state, np.ndarray):
        observable_values = [
            observable[i] * abs(np.conjugate(state[i]) * state[i])
            for i in range(2 ** nqubits)
        ]
    elif isinstance(state, dict):
        # order the counts and add zeros:
        state = dictionary_to_probabilities(state, nqubits)
        values = list(state.values())
        observable_values = [
            (observable[i] * values[i]) for i in range(2 ** nqubits)
        ]
    return sum(np.real(observable_values))


# TODO: provisional structure is below involving a user defined
# calculate_observable function as well as an observable argument used as an
# input in the calculate_observable function. Not sure this structure will
# stick.


def construct_training_data_floats(
    training_data: List[dict], observable: np.ndarray,
) -> (np.ndarray, np.ndarray):
    """Function to calculate training data now as two arrays of floats to be
    used in the regression (raw_training_data, simulated_training_data).
    Args:
        training_data: list of dictionary of counts for all training circuits
                       and all noise levels. In the form:

        ([List[dict] (simulated data), [List[List[dict]] (real circuit data)])

        observable: option to be passed to use defined observable function that
                    defines how to calculate the value of an observable from
                    the counts.
    Returns: tuple of np.ndarray of dimensions
            (num_training_circuits x noise_levels) and (num_training_circuits).
    """
    training_circuits_raw_data = training_data[1]
    training_circuits_simulated_data = training_data[0]
    noise_levels = len(training_circuits_raw_data)
    number_of_training_circuits = len(training_circuits_simulated_data)
    # first need to sort the training data, then will do a regression.
    X_data = np.zeros((number_of_training_circuits, noise_levels))
    Y_data = np.zeros((number_of_training_circuits))
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
    circuit_data: List[dict], observable: np.ndarray
) -> np.ndarray:
    """Returns circuit of interest now as two arrays of floats.
    Args:
        circuit_data: list of dictionary of counts for circuit of interest
                      and all noise levels.
        observable: option to be passed to use defined observable function that
                    defines how to calculate the value of an observable from
                    the counts.
    Returns: array of floats for observable calculated from input data.
    """
    circuit_data_floats = []
    for result in circuit_data:
        obs_raw = calculate_observable(result, observable)
        circuit_data_floats.append(obs_raw)
    return circuit_data_floats


def linear_fit_function(X_data: np.ndarray, params: List) -> float:
    """ Function used to map noisy to exact expectation values. Fitted using
    data from the training circuits.
    Args:
        X_data: array of noisy observables at various noise levels.
        *params: parameters of function.
    """
    return sum(a * x for a, x in zip(params, X_data)) + params[-1]


def dictionary_to_probabilities(
    counts: Dict[str, int], nqubits: int,
) -> Dict[str, float]:
    """Expresses the result of the simulation in the form of a dictionary
    whose values are the modulus squared of the components of the final state.
    The return probabilities are normalized by the number of counts.
    Args:
        counts: Dictionary of counts with binary keys identifying the state.
        nqubits: Number of qubits in the system.
    Returns:
        state: Dictionary whose keys are the base elements of the nqubits qubit
        system and whose values are the modulus of the corresponding squared
        amplitudes.
    """
    total_counts = sum(counts.values())
    # Initialize probabilities to 0.0
    state = {bin(j): 0.0 for j in range(2 ** nqubits)}
    # Update with normalized counts
    for key, value in counts.items():
        state[key] = value / total_counts
    return state
