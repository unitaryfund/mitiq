# Copyright (C) 2020 Unitary Fund
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

"""Contains methods used for testing mitiq's performance on random circuits."""
from typing import Tuple, Callable, Union, Optional
import numpy as np

from cirq.testing import random_circuit
from cirq import NamedQubit, Circuit

from mitiq import execute_with_zne
from mitiq._typing import QPROGRAM
from mitiq.zne.inference import Factory
from mitiq.zne.scaling import fold_gates_at_random
from mitiq.benchmarks.utils import noisy_simulation


def sample_projector(
    n_qubits: int, seed: Union[None, int, np.random.RandomState] = None,
) -> np.ndarray:
    """Constructs a projector on a random computational basis state of n_qubits.

    Args:
        n_qubits: A number of qubits
        seed: Optional seed for random number generator.
              It can be an integer or a numpy.random.RandomState object.
    Returns:
        A random computational basis projector on n_qubits. E.g., for two
        qubits this could be ``np.diag([0, 0, 0, 1])``, corresponding to the
        projector on the :math:`\\left|11\\right\\rangle` state.
    """
    obs = np.zeros(int(2 ** n_qubits))

    if seed is None:
        rnd_state = np.random
    elif isinstance(seed, int):
        rnd_state = np.random.RandomState(seed)
    else:
        rnd_state = seed

    chosenZ = rnd_state.randint(2 ** n_qubits)
    obs[chosenZ] = 1
    return np.diag(obs)


def rand_circuit_zne(
    n_qubits: int,
    depth: int,
    trials: int,
    noise: float,
    fac: Optional[Factory] = None,
    scale_noise: Callable[[QPROGRAM, float], QPROGRAM] = fold_gates_at_random,
    op_density: float = 0.99,
    silent: bool = True,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        seed: Optional seed for random number generator.

    Returns:
        The triple (exacts, unmitigateds, mitigateds) where each is a list
        whose values are the expectations of that trial in noiseless, noisy,
        and error-mitigated runs respectively.
    """
    exacts = []
    unmitigateds = []
    mitigateds = []

    qubits = [NamedQubit(str(xx)) for xx in range(n_qubits)]

    if seed:
        rnd_state = np.random.RandomState(seed)
    else:
        rnd_state = None

    for ii in range(trials):
        if not silent and ii % 10 == 0:
            print(ii)

        qc = random_circuit(
            qubits,
            n_moments=depth,
            op_density=op_density,
            random_state=rnd_state,
        )

        wvf = qc.final_wavefunction()

        # calculate the exact
        obs = sample_projector(n_qubits, seed=rnd_state)
        exact = np.conj(wvf).T @ obs @ wvf

        # make sure it is real
        exact = np.real_if_close(exact)
        assert np.isreal(exact)

        # create the simulation type
        def obs_sim(circ: Circuit) -> float:
            # we only want the expectation value not the variance
            # this is why we return [0]
            return noisy_simulation(circ, noise, obs)

        # evaluate the noisy answer
        unmitigated = obs_sim(qc)
        # evaluate the ZNE answer
        mitigated = execute_with_zne(
            qp=qc, executor=obs_sim, scale_noise=scale_noise, factory=fac
        )
        exacts.append(exact)
        unmitigateds.append(unmitigated)
        mitigateds.append(mitigated)

    return np.asarray(exacts), np.asarray(unmitigateds), np.asarray(mitigateds)
