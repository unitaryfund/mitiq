# Copyright (C) Unitary Fund
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.
"""Classical shadow estimation for quantum circuits."""

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
    observables: Optional[List[PauliString[Any]]] = None,  # type: ignore
    state_reconstruction: bool = False,
    *,
    k_shadows: Optional[int] = None,
    num_total_measurements: Optional[int] = None,
    error_rate: Optional[float] = None,
    failure_rate: Optional[float] = None,
    random_seed: Optional[int] = None,
    sampling_function_config: Dict[str, Any] = {},
) -> Dict[str, NDArray[Any]]:
    r"""
    Executes a circuit with shadow measurements.

    Args:
        circuit: The circuit to execute.
        sampling_function: The sampling function to use for z basis
            measurements. Choose from `cirq`, `qiskit`, or define your
            own sampling function.
        observables: The set of observables to measure. If None, the state
            will be reconstructed.
        state_reconstruction: Whether to reconstruct the state or estimate
            the expectation value of the observables.
        k_shadows: Number of groups of "median of means" used for shadow
            estimation.
        num_total_measurements: Number of shots per group of
            "median of means" used for shadow estimation.
        error_rate: Predicting all features with error rate \( \epsilon\)
            via median of means prediction.
        failure_rate: \( \delta\). Accurately predicting all features via
            median of means prediction with error rate less than or equals to
            \(\epsilon\) with probability at least \(1 - \delta\).
        random_seed: The random seed to use for the shadow measurements.
        sampling_function_config: A dictionary of configuration options for
            the sampling function.

    Returns:
        A dictionary containing the shadow outcomes, the Pauli strings, and
        either the estimated density matrix or the estimated expectation
        values of the observables.
    """
    # function code here

    r"""
    Executes a circuit with shadow measurements.
    Args:
        circuit: The circuit to execute.
            sampling_function: The sampling function to use for z basis
            measurements. Choose from `cirq`, `qiskit`, or define your
            own sampling function
        observables: The set of observables to measure. If None, the state
            will be reconstructed.
            state_reconstruction: Whether to reconstruct the state or estimate
            the expectation value of the observables.
        k_shadows: Number of groups of "median of means" used for shadow
            estimation
        num_total_measurements: Number of shots per group of
            "median of means" used for shadow estimation
        num_total_measurements: Total number of shots for shadow estimation
        error_rate: Predicting all features with error rate
            \( \epsilon\) via median of means prediction
        failure_rate: \( \delta\) Accurately predicting all features via
            median of means prediction with error rate less than or equals to
            \(\epsilon\) with probability at least \(1 - \delta\).
        random_seed: The random seed to use for the shadow measurements.
            sampling_function_config: A dictionary of configuration options for
            the sampling function.
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
            num_total_measurements = min_n_total_measurements(
                error_rate, num_qubits=num_qubits
            )
            k_shadows = 1
        else:  # Estimation expectation value of observables
            assert failure_rate is not None
            assert observables is not None and len(observables) > 0
            num_total_measurements, k_shadows = calculate_shadow_bound(
                error=error_rate,
                observables=observables,
                failure_rate=failure_rate,
            )
    else:
        assert num_total_measurements is not None
        if not state_reconstruction:
            assert k_shadows is not None

    if random_seed is not None:
        np.random.seed(random_seed)

    """
    Stage 1: Shadow Measurement
    """
    shadow_outcomes, pauli_strings = get_z_basis_measurement(
        circuit,
        n_total_measurements=num_total_measurements,
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
        assert k_shadows is not None
        expectation_values = [
            expectation_estimation_shadow(
                measurement_outcomes, obs, k_shadows=int(k_shadows)
            )
            for obs in observables
        ]
        output["est_observables"] = np.array(expectation_values)
    return output
