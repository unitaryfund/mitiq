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

from cirq.circuits import Circuit

from typing import List, Union, Callable, Optional

from clifford_training_data import generate_training_circuits

from data_regression import (
    scale_noise_in_circuits,
    execute_training_circuits,
    execute_circuit_of_interest,
    construct_training_data_floats,
    construct_circuit_data_floats,
    find_optimal_parameters,
    wrapper_fit_func,
    calculate_observable,
)


def execute_with_CDR(
    circuit: Circuit,
    executor: Callable[[Circuit], dict],
    simulator: Callable[[Circuit], Union[dict, np.ndarray]],
    observable: Union[List[np.ndarray], np.ndarray],
    num_training_circuits: int,
    fraction_non_clifford: float,
    scale_noise: Optional[Callable[[Circuit, float], Circuit]] = None,
    noise_scaling_factors: Optional[Union[float, List[float]]] = None,
    **kwargs: dict,
) -> (Union[float, List], Union[float, List]):
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

    Returns raw observables (at many noise levels) and mitigated observables.

    This function returns the mitigated observable/s.
    Args:
        circuit: circuit of interest compiled in the correct basis.
        executor: user defined function taking a cirq Circuit object and
                  returning a dictionary of counts.
        simulator: user defined function taking a cirq Circuit object and
                   returning either a simulated dictionary of counts or an
                   np.ndarray representing the state vector.
        observable: array of list of arrays containing the diagonal elements of
                    observable/s of interest to be mitigated. If a list is
                    passed all these observables will be mitigates with the
                    same training set.
        num_training_circuits: number of training circuits to be used in the
                               mitigation.
        fraction_non_clifford: the fraction of non-Clifford gates to be
                               subsituted in the training circuits. The higher
                               this fraction the more costly the simulations,
                               but more successful the mitigation.
        scale_noise: optional argument containing a user defined function on
                     how to increase the noise. If this argument is given then
                     the mitigation method will be vnCDR.
        noise_scaling_factors: factors by which to scale the noise, should not
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

            FITTING OPTIONS:

            - include_intercept (Bool): whether or not to include an intercept
                                        in the function mapping noisy to exact
                                        expectation values. Defult is True as
                                        in most cases it makes for a better
                                        mitigation.
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
    intercept = kwargs.get("include_intercept", True)
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
        # identify number of noise levels:
        if isinstance(noise_scaling_factors, float) or isinstance(
            noise_scaling_factors, int
        ):
            noise_levels = 2
        else:
            noise_levels = len(noise_scaling_factors) + 1

        training_circuits_list = scale_noise_in_circuits(
            training_circuits_list, scale_noise, noise_scaling_factors
        )
        # redefine circuit as a list of circuits with scaled noise:
        circuits = scale_noise_in_circuits(
            circuit, scale_noise, noise_scaling_factors
        )
    else:
        noise_levels = 1
    
    #print(len(circuits))
    # run the training circuits in the desired backend and simulator:
    results_dict_training_circuits = execute_training_circuits(
        training_circuits_list,
        executor,
        simulator,
        noise_levels,
    )
    # run the circuit of interest in the desired backend:
    results_dict_circuit_of_interest = execute_circuit_of_interest(
        circuits, executor
    )

    if isinstance(observable, list):
        #print('here')
        mitigated_observables = []
        raw_observables = []
        for obs in observable:
            circuit_data = construct_circuit_data_floats(
                results_dict_circuit_of_interest, obs, noise_levels
            )
            train_data = construct_training_data_floats(
                results_dict_training_circuits, obs, noise_levels
            )
            params = find_optimal_parameters(
                train_data[0], train_data[1], intercept=intercept
            )
            mitigated_observables.append(
                wrapper_fit_func(circuit_data, noise_levels, params, intercept)
            )
            raw_observables.append(circuit_data)
    else:
        circuit_data = construct_circuit_data_floats(
            results_dict_circuit_of_interest, observable, noise_levels
        )
        train_data = construct_training_data_floats(
            results_dict_training_circuits, observable, noise_levels
        )
        params = find_optimal_parameters(
            train_data[0], train_data[1], intercept=intercept
        )
        mitigated_observables = wrapper_fit_func(
            circuit_data, noise_levels, params, intercept
        )
        raw_observables = circuit_data
    #print(circuit_data)

    return raw_observables, mitigated_observables
