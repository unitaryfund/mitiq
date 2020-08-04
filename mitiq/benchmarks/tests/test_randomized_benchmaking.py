import pytest
from itertools import product
import numpy as np

from mitiq.benchmarks.randomized_benchmarking import rb_circuits
from mitiq.factories import (
    LinearFactory,
    RichardsonFactory,
    PolyFactory,
    ExpFactory,
    AdaExpFactory,
)
from mitiq.folding import (
    fold_gates_at_random,
    fold_gates_from_left,
    fold_gates_from_right,
    fold_global,
)
from mitiq.benchmarks.utils import noisy_simulation
from mitiq.zne import mitigate_executor

SCALE_FUNCTIONS = [
    fold_gates_at_random,
    fold_gates_from_left,
    fold_gates_from_right,
    fold_global,
]

FACTORIES = [
    AdaExpFactory(steps=3, scale_factor=1.5, asymptote=0.25),
    ExpFactory([1.0, 1.4, 2.1], asymptote=0.25),
    RichardsonFactory([1.0, 1.4, 2.1]),
    LinearFactory([1.0, 1.6]),
    PolyFactory([1.0, 1.4, 2.1], order=2),
]


def test_rb_circuits():
    depths = range(2, 10, 2)

    # test single qubit RB
    for trials in [2, 3]:
        circuits = rb_circuits(n_qubits=1, num_cliffords=depths, trials=trials)
        for qc in circuits:
            # we check the ground state population to ignore any global phase
            wvf = qc.final_wavefunction()
            zero_prob = abs(wvf[0] ** 2)
            assert np.isclose(zero_prob, 1)

    # test two qubit RB
    for trials in [2, 3]:
        circuits = rb_circuits(n_qubits=2, num_cliffords=depths, trials=trials)
        for qc in circuits:
            # we check the ground state population to ignore any global phase
            wvf = qc.final_wavefunction()
            zero_prob = abs(wvf[0] ** 2)
            assert np.isclose(zero_prob, 1)


@pytest.mark.parametrize(["scale_noise", "fac"], product(SCALE_FUNCTIONS, FACTORIES))
def test_random_benchmarks(scale_noise, fac):
    depths = [2, 4]
    trials = 3
    circuits = rb_circuits(n_qubits=2, num_cliffords=depths, trials=trials)
    noise = 0.01
    obs = np.diag([1, 0, 0, 0])

    def executor(qc):
        return noisy_simulation(qc, noise=noise, obs=obs)

    mit_executor = mitigate_executor(executor, fac, scale_noise)

    unmitigated = []
    mitigated = []
    for qc in circuits:
        unmitigated.append(executor(qc))
        mitigated.append(mit_executor(qc))

    unmit_err = np.abs(1.0 - np.asarray(unmitigated))
    mit_err = np.abs(1.0 - np.asarray(mitigated))

    assert np.average(unmit_err) >= np.average(mit_err)
