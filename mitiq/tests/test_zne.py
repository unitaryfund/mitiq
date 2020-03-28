# test_zne.py
""" This module test zne with cirq"""

import numpy as np
from typing import Tuple
import pytest

from cirq import Circuit, depolarize, LineQubit, X, DensityMatrixSimulator

from mitiq import execute_with_zne
from mitiq.factories import LinearFactory

SIMULATOR = DensityMatrixSimulator()


def meas_observable(rho: np.ndarray, obs: np.ndarray) -> Tuple[float, float]:
    """Measures a density matrix rho against observable obs.

    Args:
        rho: A density matrix.
        obs: A Hermitian observable.

    Returns:
        The tuple (expectation value, variance).
    """
    obs_avg = np.real(np.trace(rho @ obs))
    obs_delta = np.sqrt(np.real(np.trace(rho @ obs @ obs)) - obs_avg ** 2)
    return obs_avg, obs_delta


# 0.1% depolarizing noise
NOISE = 0.001


def noisy_simulation(circ: Circuit, shots=None) -> Tuple[float, float]:
    """ Simulates a circuit with depolarizing noise at level NOISE.

    Args:
        circ: The quantum program as a cirq object.
        shots: This unused parameter is needed to match mitiq's expected type
               signature for an executor function.

    Returns:
        The observable's measurements as as
        tuple (expectation value, variance).
    """
    A = np.diag([1, 0])
    circuit = circ.with_noise(depolarize(p=NOISE))
    rho = SIMULATOR.simulate(circuit).final_density_matrix
    A_avg, A_delta = meas_observable(rho, obs=A)
    return A_avg, A_delta


@pytest.mark.parametrize(["depth"], [[n] for n in range(10, 80, 20)])
def test_cirq_zne(depth):
    # This test runs circuits with an even number of X gates at varying
    # depths. All of these circuits should result in an expectation value of
    # 1 when measured in the computational basis.
    qbit = LineQubit(0)
    circ = Circuit()
    for _ in range(depth):
        assert depth % 2 == 0, "Depths must be even to ensure an " \
                               "expectation value of 1."
        circ += X(qbit)

    # We then compare the mitigated and unmitigated results.
    unmitigated, _ = noisy_simulation(circ)
    mitigated, _ = execute_with_zne(circ, noisy_simulation)
    exact = 1
    # The mitigation should improve the result.
    assert (exact - mitigated) < (exact - unmitigated)

    # Linear factories should work as well
    fac = LinearFactory([1.0, 2.0, 2.5])
    linear, _ = execute_with_zne(circ, noisy_simulation, fac=fac)
    assert (exact - linear) < (exact - unmitigated)
