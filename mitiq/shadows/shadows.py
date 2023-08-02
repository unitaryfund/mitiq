# Copyright (C) Unitary Fund
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.
"""Classical shadow estimation for quantum circuits. Based on the paper"""

from typing import Optional, Callable, List, Dict, Any, Tuple, Union, Mapping

import cirq
import numpy as np
from numpy.typing import NDArray

import mitiq
from mitiq import MeasurementResult
from mitiq.shadows.classical_postprocessing import (
    shadow_state_reconstruction,
    expectation_estimation_shadow,
    get_pauli_fidelity,
)
from mitiq.shadows.quantum_processing import random_pauli_measurement


def pauli_twirling_calibrate(
    qubits: List[cirq.Qid],
    executor: Callable[[cirq.Circuit], MeasurementResult],
    num_total_measurements_calibration: int = 50000,
    k_calibration: int = 5,
) -> Dict[str, complex]:
    r"""
    This function returns the dictionary of the median of means estimation
    of Pauli fidelity: {:math:`\{'b':f_{b}\}_{b\in\{0,1\}^n}`}.

    Args:
        qubits: The qubits to measure.
        executor: The function to use to do quantum measurement, must be same
            as executor in `shadow_quantum_processing`.
        k_calibration: Number of groups of "median of means" used for
            calibration.
        num_total_measurements_calibration: Number of shots per group of
            "median of means" used for calibration.
    Returns:
        A dictionary containing the calibration outcomes.
    """
    # calibration circuit is of same qubit number with original circuit
    zero_circuit = cirq.Circuit()
    """
    Calibration stage: calibrate errors in quantum measurement, can't
    mitigate errors in state preparation stage.
    """
    # perform random Pauli measurement one the calibration circuit
    calibration_measurement_outcomes = random_pauli_measurement(
        zero_circuit,
        n_total_measurements=num_total_measurements_calibration,
        executor=executor,
        qubits=qubits,
    )
    # get the median of means estimation of Pauli fidelities
    return get_pauli_fidelity(calibration_measurement_outcomes, k_calibration)


def shadow_quantum_processing(
    circuit: cirq.Circuit,
    executor: Callable[[cirq.Circuit], MeasurementResult],
    num_total_measurements_shadow: int,
    random_seed: Optional[int] = None,
) -> Tuple[List[str], List[str]]:
    r"""
    Executes a circuit with classical shadows. This function can be used for
    state reconstruction or expectation value estimation of observables.

    Args:
        circuit: The circuit to execute.
        executor: The function to use to do quantum measurement,
            must be same as executor in `pauli_twirling_calibrate`.
        num_total_measurements_shadow: Total number of shots for shadow
            estimation.
        random_seed: The random seed to use for the shadow measurements.

    Returns:
        A dictionary containing the bit strings, the Pauli strings
        `bit_strings`: Circuit qubits computational basis
        e.g. :math:`"01..":=|0\rangle|1\rangle..`.
        `pauli_strings`: The local Pauli measurement performed on each
        qubit. e.g."XY.." means perform local X-basis measurement on the
        1st qubit, local Y-basis measurement the 2ed qubit in the circuit.
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    r"""
    Shadow stage 1: Sample random unitary form
    :math:`\mathcal{g}\subset U(2^n)` and perform computational
    basis measurement
    """
    # random Pauli measurement on the circuit
    output = random_pauli_measurement(
        circuit,
        n_total_measurements=num_total_measurements_shadow,
        executor=executor,
    )
    return output


def classical_post_processing(
    shadow_outcomes: Tuple[List[str], List[str]],
    rshadows: bool = False,
    calibration_results: Optional[Dict[str, float]] = None,
    observables: Optional[List[mitiq.PauliString]] = None,
    k_shadows: Optional[int] = None,
    state_reconstruction: Optional[bool] = False,
) -> Mapping[str, Union[float, NDArray[Any]]]:
    r"""
    Executes a circuit with classical shadows. This function can be used for
    state reconstruction or expectation value estimation of observables.

    Args:
        shadow_outcomes: The output of function `shadow_quantum_processing`.
        rshadows: Whether to use the calibration results.
        calibration_results: The output of function `pauli_twirling_calibrate`.
        observables: The set of observables to measure.
        k_shadows: Number of groups of "median of means" used for shadow
            estimation of expectation values.
        state_reconstruction: Whether to reconstruct the state or estimate
            the expectation value of the observables.

    Returns:
        If state_reconstruction is True: state tomography matrix in
        :math:`\mathbb{M}(\mathbb{C})_{2^n}` if rshadows is False,
        otherwise state tomography vector in :math:`\mathbb{C}^{4^d}`.
        If observables is given: estimated expectation values of
        observables.
    """

    if rshadows:
        if calibration_results is None:
            raise ValueError(
                "Calibration results cannot be None when rshadows"
            )

    """
    Shadow stage 2: Estimate the expectation value of the observables OR
    reconstruct the state
    """
    output: Dict[str, Union[float, NDArray[Any]]] = {}
    if state_reconstruction:
        reconstructed_state = shadow_state_reconstruction(
            shadow_outcomes, rshadows, f_est=calibration_results
        )
        output["reconstructed_state"] = reconstructed_state  # type: ignore
    elif (
        observables is not None
    ):  # Estimation expectation value of observables
        if k_shadows is None:
            k_shadows = 1

        for obs in observables:
            expectation_values = expectation_estimation_shadow(
                shadow_outcomes,
                obs,
                k_shadows=k_shadows,
                pauli_twirling_calibration=rshadows,
                f_est=calibration_results,
            )
            output[str(obs)] = expectation_values
    return output
