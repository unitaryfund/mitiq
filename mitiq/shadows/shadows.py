# Copyright (C) Unitary Fund
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.
"""Classical shadow estimation for quantum circuits. Based on the paper"""

from typing import Optional, Callable, List, Dict, Any

import cirq
import numpy as np
from cirq.ops.pauli_string import PauliString
from numpy.typing import NDArray

from mitiq import MeasurementResult
from mitiq.shadows import (
    random_pauli_measurement,
    shadow_state_reconstruction,
    expectation_estimation_shadow,
)
from mitiq.shadows.shadows_utils import (
    min_n_total_measurements,
    calculate_shadow_bound,
)


def execute_with_shadows(
    circuit: cirq.Circuit,
    executor: Callable[[cirq.Circuit], MeasurementResult],
    observables: Optional[List[PauliString[Any]]] = None,  # type: ignore
    state_reconstruction: bool = False,
    *,
    k_shadows: Optional[int] = None,
    num_total_measurements: Optional[int] = None,
    error_rate: Optional[float] = None,
    failure_rate: Optional[float] = None,
    random_seed: Optional[int] = None,
) -> Dict[str, NDArray[Any]]:
    r"""
    Executes a circuit with classical shadows. This function can be used for
    state reconstruction or expectation value estimation of observables.

    Args:
        circuit: The circuit to execute.
        executor: The function to use to do quantum measurement.
        observables: The set of observables to measure. If None, the state
            will be reconstructed.
        state_reconstruction: Whether to reconstruct the state or estimate
            the expectation value of the observables.
        k_shadows: Number of groups of "median of means" used for shadow
            estimation.
        num_total_measurements: Number of shots per group of
            "median of means" used for shadow estimation.
        num_total_measurements: Total number of shots for shadow estimation.
        error_rate: Predicting all features with error rate
            :math:`\epsilon` via median of means prediction.
        failure_rate: :math:`\delta` Accurately predicting all features via
            median of means prediction with error rate less than or equals to
            :math:`\epsilon` with probability at least :math:`1 - \delta`.
        random_seed: The random seed to use for the shadow measurements.

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
    shadow_outcomes, pauli_strings = random_pauli_measurement(
        circuit,
        n_total_measurements=num_total_measurements,
        executor=executor,
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
