# Copyright (C) Unitary Fund
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

"""High-level probabilistic error cancellation tools."""

from typing import (
    Optional,
    Callable,
    Union,
    Sequence,
    Tuple,
    Dict,
    Any,
    cast,
    List,
)
import cirq
from functools import wraps
import warnings
import numpy as np

from mitiq import Executor, Observable, QPROGRAM, QuantumResult
from mitiq.shadows import *


shadow_config = {"RShadow":False, # Use Robust Shadow Estimation or not
                # Algorithm parameters (median of means) that users can tune (if RShadow = False, K1 and N1 are not used)
                # One can choose to fill in K1, K2, N1, N2, or leave them as None if one wants to define error_rate and precision instead.
                "K2": Optional[int],# Number of groups of "median of means" used for shadow estimation
                "N2": Optional[int],# Number of shots per group of "median of means" used for shadow estimation
                "K1": Optional[int],# Number of groups of "median of means" used for calibration in rshadow
                "N1": Optional[int],# Number of shots per group of "median of means" used for calibration in rshadow
                "error_rate": Optional[float], # epsilon
                "precision": Optional[float], # 1 - delta
                 }


def execute_with_shadows(
    circuit: cirq.Circuit,
    executor: Union[Executor, Callable],
    observables: Optional[List[Observable]] = None,
    state_reconstruction: bool = False,
    *,
    # number of shots for shadow estimation
    # w/o calibration (RShadow = False) then dim of the shadow is R_2
    # w/ calibration (RShadow = True) then dim of the shadow is R_2, and dim of calibration is R_1.
    # N1: Optional[int],
    K1: Optional[int] = None,
    R1: Optional[int] = None, # Number of Total Measurements for Classical shadow without calibration
    # N2: Optional[int],
    K2: Optional[int] = None,
    R2: Optional[int] = None, # Number of Total Measurements for calibration

    error_rate: Optional[float] = None, # epsilon
    precision: Optional[float] = None, # 1 - delta
    RShadow: Optional[bool] = False,
    random_seed: Optional[int] = None,
    max_batch_size: int = 100000000,
) -> dict:
    """
    Executes a circuit with shadow measurements.
    Args:  
        circuit: The circuit to execute.
        executor: The executor to use for running the circuit.
        observables: The observables to measure. If None, the state will be reconstructed.
        state_reconstruction: Whether to reconstruct the state or estimate the expectation value of the observables.
        K1: Number of groups of "median of means" used for calibration in rshadow
        R1: Total number of shots for calibration in rshadow
        N1: Number of shots per group of "median of means" used for calibration in rshadow N1=R1/K1
        K2: Number of groups of "median of means" used for shadow estimation
        R2: Number of shots per group of "median of means" used for shadow estimation
        N2: Total number of shots for shadow estimation N2=R2/K2
        error_rate: epsilon
        precision: 1 - delta
        RShadow: Use Robust Shadow Estimation or not
        random_seed: The random seed to use for the shadow measurements.
        max_batch_size: The maximum batch size to use for the executor.
    Returns:
        A dictionary containing the shadow outcomes, the Pauli strings, and either the estimated density matrix or the estimated expectation values of the observables.
        
    """
    
    qubits: List[cirq.GridQubit] = list(circuit.all_qubits())
    num_qubits: int = len(qubits)
    
    if observables is None:
        assert state_reconstruction is True, "observables must be provided if state_reconstruction is False"

    if error_rate is not None:
        if state_reconstruction:
            R2 = min_n_total_measurements(error_rate, num_qubits=num_qubits)
            K2 = 1
            N2 = R2
        else: # Estimation expectation value of observables
            assert precision is not None
            assert observables is not None and len(observables) > 0
            R2, K2 = shadow_bound_theorem(error=error_rate, observables=observables,failure_rate=precision)
            N2 = R2 // K2
        print('Number of total measurements: ', R2)
    else:
        assert R2 is not None
        if not state_reconstruction: 
            assert K2 is not None
            

    if random_seed is not None:
        np.random.seed(random_seed)

    if not isinstance(executor, Executor):
        assert isinstance(executor, Callable)
        executor = Executor(executor, max_batch_size=max_batch_size)



    '''
    Stage 1: Shadow Measurement
    '''
    shadow_outcomes, pauli_strings = shadow_measure_with_executor(circuit, executor, n_total_measurements=R2)
    output = {"shadow_outcomes": shadow_outcomes, "pauli_strings": pauli_strings}
    '''
    Stage 2: Estimate the expectation value of the observables OR reconstruct the state
    '''
    measurement_outcomes=(shadow_outcomes, pauli_strings)
    if state_reconstruction:
        est_density_matrix = shadow_state_reconstruction(measurement_outcomes)
        output["est_density_matrix"] = est_density_matrix
    else: # Estimation expectation value of observables
        assert observables is not None and len(observables) > 0
        expectation_values = [expectation_estimation_shadow(measurement_outcomes, obs, k=K2) for obs in observables]
        output["est_observables"] = np.array(expectation_values)
    return output





    

