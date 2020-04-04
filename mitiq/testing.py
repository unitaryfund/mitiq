# testing.py
"""
Contains methods used for testing mitiq's performance
"""
from typing import Tuple, Callable, List
import numpy as np

from cirq.testing import random_circuit
from cirq import NamedQubit, Circuit, DensityMatrixSimulator, depolarize

from mitiq import execute_with_zne, QPROGRAM
from mitiq.factories import Factory

SIMULATOR = DensityMatrixSimulator()


def sample_observable(n_qubits: int) -> np.ndarray:
    """Constructs a random computational basis observable on n_qubits

    Args:
        n_qubits: A number of qubits

    Returns:
        A random computational basis observable on n_qubits, e.g. for two
        qubits this could be np.diag([0, 0, 0, 1]) to measure the ZZ
        observable.
    """
    obs = np.zeros(int(2 ** n_qubits))
    chosenZ = np.random.randint(2 ** n_qubits)
    obs[chosenZ] = 1
    return np.diag(obs)


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


def noisy_simulation(circ: Circuit, obs: np.ndarray, noise: float) -> \
        Tuple[float, float]:
    """ Simulates a circuit with depolarizing noise at level NOISE.

    Args:
        circ: The quantum program as a cirq object.
        obs: The observable for the simulation to measure.
        noise: The noise level of the simulation for a depolarizing channel.

    Returns:
        The observable's measurements as as
        tuple (expectation value, variance).
    """
    circuit = circ.with_noise(depolarize(p=noise))
    rho = SIMULATOR.simulate(circuit).final_density_matrix
    A_avg, A_delta = meas_observable(rho, obs=obs)
    return A_avg, A_delta


def rand_benchmark_zne(n_qubits: int, depth: int, trials: int, noise: float,
                       fac: Factory=None,
                       scale_noise: Callable[[QPROGRAM, float], QPROGRAM]=None,
                       op_density:float=0.99, silent:bool=True) \
        -> Tuple[List, List]:
    """Benchmarks a zero-noise extrapolation method and noise scaling executor
    by running on randomly sampled quantum circuits.

    Args:
        n_qubits: The number of qubits.
        depth: The depth in moments of the random circuits.
        trials: The number of random circuits to average over.
        noise: The noise level of the depolarizing channel for simulation.
        fac: The Factory giving the extrapolation method.
        scale_noise: The method for scaling noise, e.g. fold_gates_at_random
        op_density: The expected proportion of qubits that are acted on in
                    any moment.
        silent: If False will print out statements every tenth trial to
                track progress.

    Returns:
        The tuple (unmitigated_error, mitigated_error) where each is a list
        whose values are the errors of that trial in the unmitigated or
        mitigated cases.
    """
    unmitigated_error = []
    mitigated_error = []

    for ii in range(trials):
        if not silent and ii % 10 == 0: print(ii)

        qubits = [NamedQubit(str(xx)) for xx in range(n_qubits)]
        qc = random_circuit(qubits, n_moments=depth, op_density=op_density)
        wvf = qc.final_wavefunction()

        # calculate the exact
        obs = sample_observable(n_qubits)
        exact = np.conj(wvf).T @ obs @ wvf

        # make sure it is real
        exact = np.real_if_close(exact)
        assert np.isreal(exact)

        # create the simulation type
        def obs_sim(circ: Circuit, shots=None):
            return noisy_simulation(circ, obs, noise)

        # evaluate the noisy answer
        unmitigated, _ = obs_sim(qc)
        # evaluate the ZNE answer
        mitigated, _ = execute_with_zne(qp=qc, executor=obs_sim,
                                        scale_noise=scale_noise,
                                        fac=fac)
        # We are going to resuse the same factory on the next step of the loop
        # and so we need to reset its instack and outstack values.
        fac.reset()

        unmitigated_error.append(np.abs(exact - unmitigated))
        mitigated_error.append(np.abs(exact - mitigated))

    return unmitigated_error, mitigated_error
