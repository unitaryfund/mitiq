# random_circ.py
"""
Contains methods used for testing mitiq's performance
"""
from typing import Tuple, Callable, List
import numpy as np

from cirq.testing import random_circuit
from cirq import NamedQubit, Circuit

from mitiq import execute_with_zne, QPROGRAM
from mitiq.factories import Factory
from mitiq.benchmarks.utils import noisy_simulation


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

    qubits = [NamedQubit(str(xx)) for xx in range(n_qubits)]

    for ii in range(trials):
        if not silent and ii % 10 == 0: print(ii)

        qc = random_circuit(qubits, n_moments=depth, op_density=op_density)
        wvf = qc.final_wavefunction()

        # calculate the exact
        obs = np.diag([1 / 2**n_qubits] * 2**n_qubits)
        exact = np.conj(wvf).T @ obs @ wvf

        # make sure it is real
        exact = np.real_if_close(exact)
        assert np.isreal(exact)

        # fixes the noise level and the observable
        def obs_sim(circ: Circuit):
            return noisy_simulation(circ, noise, obs)

        # evaluate the noisy answer
        unmitigated = obs_sim(qc)
        # evaluate the ZNE answer
        mitigated = execute_with_zne(qp=qc, executor=obs_sim,
                                        scale_noise=scale_noise,
                                        fac=fac)

        unmitigated_error.append(np.abs(exact - unmitigated))
        mitigated_error.append(np.abs(exact - mitigated))

    return unmitigated_error, mitigated_error
