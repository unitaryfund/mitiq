# test_zne.py
"""This module tests zne with Cirq."""

import numpy as np
import pytest
from itertools import product

from cirq import Circuit, depolarize, LineQubit, X, DensityMatrixSimulator

from mitiq import execute_with_zne, mitigate_executor
from mitiq.factories import LinearFactory, RichardsonFactory, PolyFactory
from mitiq.folding import fold_gates_at_random, fold_gates_from_left, \
    fold_gates_from_right
from mitiq.testing import rand_benchmark_zne

SIMULATOR = DensityMatrixSimulator()

# 0.1% depolarizing noise
NOISE = 0.001


def noisy_simulation(circ: Circuit, shots=None) -> float:
    """ Simulates a circuit with depolarizing noise at level NOISE.

    Args:
        circ: The quantum program as a cirq object.
        shots: This unused parameter is needed to match mitiq's expected type
               signature for an executor function.

    Returns:
        The observable's measurements as as
        tuple (expectation value, variance).
    """
    circuit = circ.with_noise(depolarize(p=NOISE))
    rho = SIMULATOR.simulate(circuit).final_density_matrix
    # define the computational basis observable
    obs = np.diag([1, 0])
    expectation = np.real(np.trace(rho @ obs))
    return expectation

# This test is somewhat redundant with the random circuit tests but we keep in
# because this tests the exact code used in the getting started.
@pytest.mark.parametrize(["depth"], [[n] for n in range(10, 80, 20)])
def test_cirq_zne(depth):
    # This test runs circuits with an even number of X gates at varying
    # depths. All of these circuits should result in an expectation value of
    # 1 when measured in the computational basis.
    qbit = LineQubit(0)
    assert depth % 2 == 0, "Depths must be even to ensure an " \
                           "expectation value of 1."
    circ = Circuit(X(qbit) for _ in range(depth))

    # We then compare the mitigated and unmitigated results.
    unmitigated = noisy_simulation(circ)
    mitigated = execute_with_zne(circ, noisy_simulation)
    exact = 1
    # The mitigation should improve the result.
    assert abs(exact - mitigated) < abs(exact - unmitigated)

    # Linear factories should work as well
    fac = LinearFactory(scalars=[1.0, 2.0, 2.5])
    linear = execute_with_zne(circ, noisy_simulation, fac=fac)
    assert abs(exact - linear) < abs(exact - unmitigated)

    # Test the mitigate executor
    run_mitigated = mitigate_executor(noisy_simulation)
    e_mitigated = run_mitigated(circ)
    assert np.isclose(e_mitigated, mitigated)


SCALE_FUNCTIONS = [
    fold_gates_at_random,
    fold_gates_from_left,
    fold_gates_from_right
]

FACTORIES = [
    RichardsonFactory([1.0, 1.4, 2.1]),
    LinearFactory([1.0, 1.6]),
    PolyFactory([1.0, 1.4, 2.1], order=2)
]


@pytest.mark.parametrize(["scale_noise", "fac"],
                         product(SCALE_FUNCTIONS, FACTORIES))
def test_random_benchmarks(scale_noise, fac):
    unmit_err, mit_err = rand_benchmark_zne(n_qubits=2,
                                            depth=20,
                                            trials=8,
                                            op_density=0.99,
                                            noise=0.003,
                                            fac=fac,
                                            scale_noise=scale_noise)

    assert np.average(unmit_err) >= np.average(mit_err)
