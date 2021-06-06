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

"""API for using Clifford Data Regression (CDR) error mitigation."""

from typing import List, Union, Callable, Sequence

import numpy as np
from scipy.optimize import curve_fit

from cirq.circuits import Circuit

from mitiq.cdr.clifford_training_data import generate_training_circuits
from mitiq.cdr.data_regression import linear_fit_function
from mitiq.cdr.execute import calculate_observable
from mitiq.zne.scaling import fold_gates_at_random


# TODO: Allow for any QPROGRAM, not just a cirq.Circuit.
def execute_with_CDR(
    circuit: Circuit,
    executor: Callable[[Circuit], dict],
    simulator: Callable[[Circuit], Union[dict, np.ndarray]],
    observables: List[np.ndarray],
    num_training_circuits: int,
    fraction_non_clifford: float,
    ansatz: Callable[..., float] = linear_fit_function,
    num_parameters: int = None,
    scale_factors: Sequence[float] = (1,),
    scale_noise: Callable[[Circuit, float], Circuit] = fold_gates_at_random,
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

    Returns list of raw observables (at noise scale factors) and mitigated
    observables.

    This function returns the mitigated observable/s.
    Args:
        circuit: Circuit of interest compiled in the correct basis.
        executor: User defined function taking a cirq Circuit object and
                  returning a dictionary of counts.
        simulator: User defined function taking a cirq Circuit object and
                   returning either a simulated dictionary of counts or an
                   np.ndarray representing the state vector.
        observables: List of arrays containing the diagonal elements of
                    observable/s of interest to be mitigated. If a list is
                    passed all these observables will be mitigates with the
                    same training set.
        num_training_circuits: Number of training circuits to be used in the
                               mitigation.
        fraction_non_clifford: The fraction of non-Clifford gates to be
                               subsituted in the training circuits. The higher
                               this fraction the more costly the simulations,
                               but more successful the mitigation.
        ansatz: The function to map noisy to exact data. Takes array of noisy
                and data and parameters returning a float.
        num_parameters: The number of paramters the ansatz takes.
        scale_noise: Optional argument containing a user defined function on
                     how to increase the noise. If this argument is given then
                     the mitigation method will be vnCDR.
        scale_factors: Factors by which to scale the noise, should not
                               include 1 as this is just the original circuit.
        kwargs: Available keyword arguments are:

        TRAINING SET CONSTRUCTION OPTIONS:

            - method_select (string): Specifies the method used to select the
                                      non-Clifford gates to replace when
                                      constructing the near-Clifford training
                                      circuits. Available options are:
                                            ['uniform', 'gaussian']
            - method_replace (string): Specifies the method used to replace the
                                      selected non-Clifford gates with a
                                      Clifford when constructing the
                                      near-Clifford training circuits.
                                      Available options are:
                                        ['uniform', 'gaussian', 'closest']
            - sigma_select (float): Width of the Gaussian distribution used for
                                    ``method_select='gaussian'``.
            - sigma_replace (float): Width of the Gaussian distribution used
                                     for ``method_replace='gaussian'``.
            - random_state (int): Seed for sampling.

    Returns: The tuple (raw_expectations, mitigated_expectations)
             corresponding to the many raw expectation values (at different
             noise levels) and the associated mitigated values.

    .. [Czarnik2020] : Piotr Czarnik, Andrew Arramsmith, Patrick Coles,
        Lukasz Cincio, "Error mitigation with Clifford quantum circuit
        data," (https://arxiv.org/abs/2005.10189).
    .. [Lowe2020] : Angus Lowe, Max Hunter Gordon, Piotr Czarnik,
        Andrew Arramsmith, Patrick Coles, Lukasz Cincio,
        "Unified approach to data-driven error mitigation,"
        (https://arxiv.org/abs/2011.01157).
    """
    # Handle keyword arguments for generating training circuits.
    method_select = kwargs.get("method_select", "uniform")
    method_replace = kwargs.get("method_replace", "closest")
    random_state = kwargs.get("random_state", None)
    kwargs_for_training_set_generation = {
        "sigma_select": kwargs.get("sigma_select"),
        "sigma_replace": kwargs.get("sigma_replace"),
    }

    # Generate training circuits.
    training_circuits = generate_training_circuits(
        circuit,
        num_training_circuits,
        fraction_non_clifford,
        method_select,
        method_replace,
        random_state,
        kwargs=kwargs_for_training_set_generation,
    )

    # [Optionally] Scale noise in circuits.
    all_circuits = [
        [scale_noise(c, s) for s in scale_factors]
        for c in [circuit] + training_circuits
    ]

    # Execute all circuits to get MeasurementResult's. TODO: Allow batching.
    noisy_counts = np.array(
        [[executor(circ) for circ in circuits] for circuits in all_circuits]
    )
    ideal_counts = np.array([simulator(circ) for circ in all_circuits[0]])

    # Do the regression.
    results_dict_circuit_of_interest = noisy_counts[:, 0]

    mitigated_observables = []
    raw_observables = []
    for obs in observables:
        circuit_data = np.array([
            calculate_observable(state_or_measurements=measurements, observable=obs)
            for measurements in results_dict_circuit_of_interest
        ])

        # Get the noisy ⟨𝛹| O |𝛹⟩ from the noisy (executor) counts.
        noisy_expectation_values = np.array([
            [calculate_observable(state_or_measurements=measurements, observable=obs) for measurements in row]
            for row in noisy_counts
        ])

        # Get the exact ⟨𝛹| O |𝛹⟩ from the exact (simulator) counts.
        ideal_expectation_values = np.array([
            calculate_observable(state_or_measurements=measurements, observable=obs) for measurements in ideal_counts
        ])

        # Do the regression.
        fitted_params, _ = curve_fit(
            lambda x, *params: ansatz(x, params),
            noisy_expectation_values,
            ideal_expectation_values,
            p0=np.zeros(num_parameters),
        )
        mitigated_observables.append(ansatz(circuit_data, fitted_params))
        raw_observables.append(circuit_data)

    return raw_observables, mitigated_observables
