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

""" Functions for execution of CDR mitigation on circuit of interest"""

import numpy as np

from scipy.optimize import curve_fit

from cirq.circuits import Circuit

from typing import List, Union, Callable, Optional

from mitiq.cdr.clifford_training_data import generate_training_circuits

from mitiq.cdr.data_regression import (
    scale_noise_in_circuits,
    construct_training_data_floats,
    construct_circuit_data_floats,
    linear_fit_function,
)


def execute_with_CDR(
    circuit: Circuit,
    executor: Callable[[Circuit], dict],
    simulator: Callable[[Circuit], Union[dict, np.ndarray]],
    observables: List[np.ndarray],
    num_training_circuits: int,
    fraction_non_clifford: float,
    ansatz: Callable[[np.ndarray, List], List] = linear_fit_function,
    num_parameters: int = None,
    scale_noise: Optional[Callable[[Circuit, float], Circuit]] = None,
    scale_factors: Optional[List[float]] = None,
    **kwargs: dict,
) -> (List[List], List[List]):
    """Function for the calculation of an observable from some circuit of
    interest to be mitigated with CDR (or vnCDR) based on [Czarnik2020]_ and
    [Lowe2020]_.

    The circuit of interest must be compiled in the native basis of the IBM
    quantum computers, that is {Rz, sqrt(X), CNOT}, or such that all the
    non-Clifford gates are contained in the Rz rotations.

    The observable/s to be calculated should be input as an array or a list of
    arrays representing the diagonal of the observables to be measured. Note
    these observables MUST be diagonal in z-basis measurements corresponding to
    the circuit of interest.

    Returns list of raw observables (at many noise levels) and mitigated
    observables.

    This function returns the mitigated observable/s.
    Args:
        circuit: circuit of interest compiled in the correct basis.
        executor: user defined function taking a cirq Circuit object and
                  returning a dictionary of counts.
        simulator: user defined function taking a cirq Circuit object and
                   returning either a simulated dictionary of counts or an
                   np.ndarray representing the state vector.
        observable: list of arrays containing the diagonal elements of
                    observable/s of interest to be mitigated. If a list is
                    passed all these observables will be mitigates with the
                    same training set.
        num_training_circuits: number of training circuits to be used in the
                               mitigation.
        fraction_non_clifford: the fraction of non-Clifford gates to be
                               subsituted in the training circuits. The higher
                               this fraction the more costly the simulations,
                               but more successful the mitigation.
        ansatz: the function to map noisy to exact data. Takes array of noisy
                and data and parameters returning a float.
        num_parameters: the number of paramters the ansatz takes.
        scale_noise: optional argument containing a user defined function on
                     how to increase the noise. If this argument is given then
                     the mitigation method will be vnCDR.
        scale_factors: factors by which to scale the noise, should not
                               include 1 as this is just the original circuit.
        kwargs: Available keyword arguments are:

            TRAINING SET CONSTRUCTION OPTIONS:

            - method_select (string): specifies the method used to select the
                                      non-Clifford gates to replace when
                                      constructing the near-Clifford training
                                      circuits. Available options are:
                                            ['uniform', 'gaussian']
            - method_replace (string): specifies the method used to replace the
                                      selected non-Clifford gates with a
                                      Clifford when constructing the
                                      near-Clifford training circuits.
                                      Available options are:
                                        ['uniform', 'gaussian', 'closest']
            - sigma_select (float): Width of the Gaussian distribution used for
                                    ``method_select='gaussian'``.
            - sigma_replace (float): Width of the Gaussian distribution used
                                     for ``method_replace='gaussian'``.
            - random_state (int): seed for sampling.
    Returns: The tuple (raw_expectations, mitigated_expectations)
             corresponding to the many raw expectation values (at different
             noise levels) and the associated mitigated values.

    .. [Czarnik2020] : Piotr Czarnik, Andrew Arramsmith, Patrick Coles,
        Lukasz Cincio, "Error mitigation with Clifford quantum circuit
        data," (https://arxiv.org/abs/2005.10189).
    .. [Lowe2020] : Angus Lowe, Max Hunter Gordon, Piotr Czarnik,
        Andrew Arramsmith, Patrick Coles, Lukasz Cincio,
        "Unified approach to data-driven error mitigation,"
        (https://arxiv.org/abs/2011.01157)."""
    # Extracting kwargs:
    method_select = kwargs.get("method_select", "uniform")
    method_replace = kwargs.get("method_replace", "closest")
    random_state = kwargs.get("random_state", None)
    training_set_generation_kwargs_keys = ["sigma_select", "sigma_replace"]
    kwargs_for_training_set_generation = {
        key: kwargs.get(key) for key in training_set_generation_kwargs_keys
    }
    training_circuits_list = generate_training_circuits(
        circuit,
        num_training_circuits,
        fraction_non_clifford,
        method_select,
        method_replace,
        random_state,
        kwargs=kwargs_for_training_set_generation,
    )
    # If specified scale the noise in the circuit of interest and training
    # circuits:
    if scale_noise:
        # to define number of paramters in defult linear model:
        if not num_parameters:
            num_parameters = len(scale_factors) + 1
        training_circuits_list = scale_noise_in_circuits(
            training_circuits_list, scale_noise, scale_factors
        )
        # redefine circuit as a list of circuits with scaled noise:
        circuits = scale_noise_in_circuits(
            [circuit], scale_noise, scale_factors
        )
    else:
        # to define number of parameters in defult linear model:
        if not num_parameters:
            num_parameters = 2
        # both need to be list of lists of circuits:
        circuits = [[circuit]]
        training_circuits_list = [training_circuits_list]
    # execute the training circuits and the circuit of interest:
    # list to store training circuits executed with hardware:
    training_circuits_raw_data = [
        [] for i in range(len(training_circuits_list))
    ]
    # list to store simulated training circuits:
    training_circuits_simulated_data = []
    for i, training_circuits in enumerate(training_circuits_list):
        for j, circuit in enumerate(training_circuits):
            training_circuits_raw_data[i].append(executor(circuit))
            # runs the circuits with no increased noise in the simulator:
            if i == 0:
                training_circuits_simulated_data.append(simulator(circuit))
    results_dict_training_circuits = [
        training_circuits_raw_data,
        training_circuits_simulated_data,
    ]
    # circuit of interest execution:
    results_dict_circuit_of_interest = []
    for circuit in circuits:
        circuit_raw_result = executor(circuit[0])
        results_dict_circuit_of_interest.append(circuit_raw_result)
    # Now the regression:
    mitigated_observables = []
    raw_observables = []
    for obs in observables:
        circuit_data = construct_circuit_data_floats(
            results_dict_circuit_of_interest, obs
        )
        train_data = construct_training_data_floats(
            results_dict_training_circuits, obs
        )
        # going to add general regression section here:
        initial_params = np.zeros((num_parameters))
        fitted_params, _ = curve_fit(
            lambda x, *params: ansatz(x, params),
            train_data[0].T,
            train_data[1],
            p0=initial_params,
        )
        mitigated_observables.append(ansatz(circuit_data, fitted_params))
        raw_observables.append(circuit_data)

    return raw_observables, mitigated_observables
