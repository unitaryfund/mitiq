# Copyright (C) Unitary Fund
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.
"""High-level probabilistic error cancellation tools."""

from typing import Optional, Callable, Union, List, Dict, Any

import cirq
import numpy as np
from numpy.typing import NDArray
from cirq.ops.pauli_string import PauliString

from mitiq import MeasurementResult
from mitiq.shadows import (
    get_z_basis_measurement,
    shadow_state_reconstruction,
    expectation_estimation_shadow,
)
from mitiq.shadows.shadows_utils import (
    min_n_total_measurements,
    calculate_shadow_bound,
)


def execute_with_shadows(
    circuit: cirq.Circuit,
    sampling_function: Union[str, Callable[..., MeasurementResult]] = "cirq",
    # choose from cirq, qiskit, or define your own sampling function
    observables: Optional[List[PauliString[Any]]] = None,  # type: ignore
    state_reconstruction: bool = False,
    RShadow: Optional[bool] = False,
    *,
    # number of shots for shadow estimation
    # w/o calibration (RShadow = False) then dim of the shadow is R_2
    # w/ calibration (RShadow = True) then dim of the shadow is R_2,
    # and dim of calibration is R_1.
    estimation_total_rounds: Optional[int] = None,
    # Number of Total Measurements for Classical shadow without calibration
    K2: Optional[int] = None,
    measurement_total_rounds: Optional[
        int
    ] = None,  # Number of Total Measurements for calibration
    error_rate: Optional[float] = None,  # epsilon
    precision: Optional[float] = None,  # 1 - delta
    random_seed: int = 0,
    sampling_function_config: Dict[str, Any] = {},
) -> Dict[str, NDArray[Any]]:
    """
    Executes a circuit with shadow measurements.
    Args:
        circuit: The circuit to execute.
        sampling_function: The sampling function to use for z basis
            measurements.
        observables: The observables to measure. If None, the state will be
            reconstructed.
        state_reconstruction: Whether to reconstruct the state or estimate the
            expectation value of the observables.
        K1: Number of groups of "median of means" used for calibration in
            rshadow
        estimation_total_rounds: Total number of shots for calibration in
            rshadow
        N1: Number of shots per group of "median of means" used for calibration
            in rshadow N1=estimation_total_rounds/K1
        K2: Number of groups of "median of means" used for shadow estimation
            measurement_total_rounds: Number of shots per group of
             "median of means" used for shadow estimation
        N2: Total number of shots for shadow estimation
            N2=measurement_total_rounds/K2
        error_rate: epsilon
        precision: 1 - delta
        RShadow: Use Robust Shadow Estimation or not
        random_seed: The random seed to use for the shadow measurements.
        sampling_function_config: A dictionary of configuration options for the
            sampling function.
    Returns:
        A dictionary containing the shadow outcomes, the Pauli strings, and
        either the estimated density matrix or the estimated expectation
        values of the observables.
    """

    qubits = list(circuit.all_qubits())
    num_qubits = len(qubits)

    if observables is None:
        assert (
            state_reconstruction is True
        ), "observables must be provided if state_reconstruction is False"

    if error_rate is not None:
        if state_reconstruction:
            measurement_total_rounds = min_n_total_measurements(
                error_rate, num_qubits=num_qubits
            )
            K2 = 1
        else:  # Estimation expectation value of observables
            assert precision is not None
            assert observables is not None and len(observables) > 0
            measurement_total_rounds, K2 = calculate_shadow_bound(
                error=error_rate,
                observables=observables,
                failure_rate=precision,
            )
        print("Number of total measurements: ", measurement_total_rounds)
    else:
        assert measurement_total_rounds is not None
        if not state_reconstruction:
            assert K2 is not None

    if random_seed is not None:
        np.random.seed(random_seed)

    """
    Stage 1: Shadow Measurement
    """
    shadow_outcomes, pauli_strings = get_z_basis_measurement(
        circuit,
        n_total_measurements=measurement_total_rounds,
        sampling_function=sampling_function,
        sampling_function_config=sampling_function_config,
    )
    output = {
        "shadow_outcomes": shadow_outcomes,
        "pauli_strings": pauli_strings,
    }
    """
    Stage 2: Estimate the expectation value of the observables OR reconstruct
    the state
    """
    measurement_outcomes = (shadow_outcomes, pauli_strings)
    if state_reconstruction:
        est_density_matrix = shadow_state_reconstruction(measurement_outcomes)
        output["est_density_matrix"] = est_density_matrix
    else:  # Estimation expectation value of observables
        assert observables is not None and len(observables) > 0
        assert K2 is not None
        expectation_values = [
            expectation_estimation_shadow(measurement_outcomes, obs, k=int(K2))
            for obs in observables
        ]
        output["est_observables"] = np.array(expectation_values)
    return output
