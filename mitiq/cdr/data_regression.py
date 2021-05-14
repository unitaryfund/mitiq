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

""" Functions for using near-Clifford training data for error mitiaiton"""
from typing import List, Union, Callable, Dict

import numpy as np
from mitiq._typing import QPROGRAM

from cirq.circuits import Circuit

from scipy.optimize import curve_fit


def scale_noise_in_circuits(
    circuits: Union[Circuit, List[Circuit]],
    scale_noise: Callable[[Circuit, float], Circuit],
    scale_factor: Union[float, List],
) -> List[Circuit]:
    """Returns a list consisting of the original circuit followed by the circuit
    folded by the specified method and the scale factors.

    Args:
        circuits: list of training circuits.
        scale_factor: factor by which to scale the circuits.
        scale_noise: method to use to fold the circuits.
    """
    if isinstance(circuits, Circuit):
        circuits = [circuits]
    folded_circuits = []
    if isinstance(scale_factor, float) or isinstance(scale_factor, int):
        scale_factor = [scale_factor]
    for scale_factor_ in scale_factor:
        for circuit in circuits:
            folded_circuits.append(scale_noise(circuit, scale_factor_))
    return circuits + folded_circuits


def execute_training_circuits(
    all_training_circuits: List[Circuit],
    executor: Callable[[QPROGRAM], Dict],
    simulator: Callable[[QPROGRAM], Union[Dict, np.ndarray]],
    noise_levels: int = 1,
) -> (List[Dict], List[Dict]):
    """Returns two list of dictionaries of different sizes. One list of
    dictionaries consisting of the real counts at various noise levels for the
    training circuits and the other contains the exact counts.

    Args:
        all_training_circuits: list of all training.
        executor: user defined executor function.
        simulator: user definied simulator function either returns counts or
                   a statevector.
        noise_levels: number of noise levels to be used in the regression.
    """
    number_of_training_circuits = int(
        len(all_training_circuits) / noise_levels
    )
    training_circuits_raw_data = []
    training_circuits_simulated_data = []
    for i, training_circuit in enumerate(all_training_circuits):
        training_circuit_raw_result = executor(training_circuit)
        training_circuits_raw_data.append(training_circuit_raw_result)
        # runs the circuits with no increased noise in the simulator:
        if i < number_of_training_circuits:
            training_circuit_simualted_result = simulator(training_circuit)
            training_circuits_simulated_data.append(
                training_circuit_simualted_result
            )
    return (training_circuits_raw_data, training_circuits_simulated_data)


# TODO: this function below is not really neccesary, but is quite useful and
# follows the structure of the above format.
def execute_circuit_of_interest(
    circuits: Union[List[Circuit], Circuit],
    executor: Callable[[QPROGRAM], Dict],
) -> List[Dict]:
    """Function to run circuit of interest at different noise levels. Returns
    list of dictionaries of counts at different noise levels. Output to be
    passed into regression functions.

    Args:
        circuits: list of circuit of interest at different noise levels.
        executor: user defined executor function.
    """
    circuits_raw_data = []
    if isinstance(circuits, List):
        for circuit in circuits:
            circuit_raw_result = executor(circuit)
            circuits_raw_data.append(circuit_raw_result)
    else:
        circuit_raw_result = executor(circuits)
        circuits_raw_data.append(circuit_raw_result)
    return circuits_raw_data


def calculate_observable(
    state: Union[Dict, np.ndarray], observable: np.ndarray
) -> float:
    """ Function to take a users definition of an observable and calculate its
    value from the counts passed into the function (counts could be simulated
    or from raw data).
    Args:
        state: the state represented in raw or simulated counts with which to
               extract the observable or as a statevector.
        observable: array of diagonal elements of observable to be measured,
                    which is a diagonal matrix.
    """
    Q = int(np.log2(np.shape(observable)[0]))
    if isinstance(state, np.ndarray):
        observable_values = [
            observable[i] * abs(np.conjugate(state[i]) * state[i])
            for i in range(2 ** Q)
        ]
    elif isinstance(state, Dict):
        # order the counts and add zeros:
        state = state_counts(state, Q)
        values = list(state.values())
        observable_values = [
            (observable[i] * values[i]) for i in range(2 ** Q)
        ]
    return sum(np.real(observable_values))


# TODO: provisional structure is below involving a user defined
# calculate_observable function as well as an observable argument used as an
# input in the calculate_observable function. Not sure this structure will
# stick.


def construct_training_data_floats(
    training_data: List[Dict], observable: np.ndarray, noise_levels: int = 1,
) -> (np.ndarray, np.ndarray):
    """Returns training data now as two arrays of floats to be used in the
    regression (raw_training_data, simualated_training_data)
    Args:
        training_data: list of dictionary of counts for all training circuits
                       and all noise levels. In the form:

            [List[Dict] (real circuit data), List[Dict] (simulated data)]

        observable: option to be passed to use defined observable function that
                    defines how to calculate the value of an observable from
                    the counts.
        noise_levels: number of noise levels to be used in the regression.
    """
    training_circuits_raw_data = training_data[0]
    training_circuits_simualted_data = training_data[1]
    number_of_training_circuits = len(training_circuits_simualted_data)
    # first need to sort the training data, then will do a regression.
    X_data = np.zeros((number_of_training_circuits, noise_levels))
    Y_data = np.zeros((number_of_training_circuits))
    for count in range(len(training_circuits_raw_data)):
        i = count % number_of_training_circuits  # row: training cirucit
        j = int(count / number_of_training_circuits)  # column: noise levels
        training_circ_raw_data = training_circuits_raw_data[count]
        training_obs_raw = calculate_observable(
            training_circ_raw_data, observable
        )
        X_data[i, j] = training_obs_raw
        if count < number_of_training_circuits:
            training_circ_simulated_data = training_circuits_simualted_data[
                count
            ]
            training_obs_sim = calculate_observable(
                training_circ_simulated_data, observable
            )
            Y_data[count] = training_obs_sim
    return (X_data, Y_data)


# TODO: again this function below is not really neccesary, but is quite useful
#  and follows the structure of the above format.


def construct_circuit_data_floats(
    circuit_data: List[Dict], observable: np.ndarray, noise_levels: int = 1,
) -> np.ndarray:
    """Returns training data now as two arrays of floats to be used in the
    regression (raw_training_data, simualated_training_data)
    Args:
        circuit_data: list of dictionary of counts for circuit of interest
                      and all noise levels.
        observable: option to be passed to use defined observable function that
                    defines how to calculate the value of an observable from
                    the counts.
        noise_levels: number of noise levels to be used in the regression.
    """
    circuit_data_floats = []
    for result in circuit_data:
        obs_raw = calculate_observable(result, observable)
        circuit_data_floats.append(obs_raw)
    return circuit_data_floats


def find_optimal_parameters(
    X_data: np.ndarray, y_data: np.ndarray, intercept: bool = True
):
    """Returns the optimal parameters calculcated by computing a regression
    on the training data.
    Args:
        X_data: array of noisy observables at various noise levels.
        y_data: array of exact observables.
        intercept: whether or not to include and intercept in the fitting
                   function.
    """
    noise_levels = len(X_data[0][:])
    X_data = X_data.T
    if intercept:
        params_0 = np.zeros((noise_levels + 1))
    else:
        params_0 = np.zeros((noise_levels))

    # Here need to use some package that does the regression.
    popt, pcov = curve_fit(
        lambda x, *params: wrapper_fit_func(
            x, noise_levels, params, intercept
        ),
        X_data,
        y_data,
        p0=params_0,
    )
    return popt


def f_intercept(X_data: np.ndarray, params: List) -> float:
    """ Function used to map noisy to exact expectation values. Fitted using
    data from the training circuits.
    Args:
        X_data: array of noisy observables are various noise levels.
        *params: parameters of function.
    """
    return sum(a * x for a, x in zip(params, X_data)) + params[-1]


def f(X_data: np.ndarray, params: np.ndarray) -> float:
    """ Function used to map noisy to exact expectation values. Fitted using
    data from the training circuits. No intercept.
    Args:
        X_data: array of noisy observables are various noise levels.
        *params: parameters of function.
    """
    return sum(a * x for a, x in zip(params, X_data))


def wrapper_fit_func(x, noise_levels, *args, intercept: bool = True):
    if intercept:
        a = list(args[0][: noise_levels + 1])
        return f_intercept(x, a)
    else:
        a = list(args[0][:noise_levels])
        return f(x, a)


# TODO: Discuss if this format for converting shots makes sense in general.
def state_counts(counts: Dict, Q: int) -> Dict:
    """Expresses the result of the simulation in the form of a dictionary
    whose values are the modulus squared of the components of the final state.
    The return probabilities are normalised by the number of counts.
    Args:
        counts: Dictionary of counts with binary keys identifying the state.
        Q: Number of qubits in the system.
    Returns:
        state: Dictionary whose keys are the base elements of the Q qubit
        system and whose values are the modulus of the corresponding squared
        amplitudes.
    """
    basis = {i: bin(i) for i in range(2 ** Q)}
    counts_order = np.array([i for i in range(2 ** Q)])
    for i in range(len(basis)):
        key = list(basis.values())[i]
        if key not in counts:
            counts[key] = 0
        for j in range(len(counts)):
            if list(basis.values())[i] == list(counts.keys())[j]:
                counts_order[i] = list(counts.values())[j]
    state = {
        list(basis.values())[i]: (counts_order[i] / sum(counts_order))
        for i in range(2 ** Q)
    }
    return state
