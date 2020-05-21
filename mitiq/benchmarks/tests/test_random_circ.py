from itertools import product
import pytest

import numpy as np

from mitiq.factories import LinearFactory, RichardsonFactory, PolyFactory
from mitiq.folding import fold_gates_at_random, fold_gates_from_left, \
    fold_gates_from_right
from mitiq.benchmarks.random_circ import rand_benchmark_zne, sample_projector

# Fix a seed for this test file
SEED = 808


# Make fold_gates_at_random deterministic
def fold_gates_at_random_seeded(circuit, scale_factor):
    return fold_gates_at_random(circuit, scale_factor, seed=SEED)


SCALE_FUNCTIONS = [
    fold_gates_at_random_seeded,
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

    exact, unmitigated, mitigated = rand_benchmark_zne(n_qubits=2,
                                                       depth=20,
                                                       trials=8,
                                                       op_density=0.99,
                                                       noise=0.003,
                                                       fac=fac,
                                                       scale_noise=scale_noise,
                                                       seed=SEED)

    unmit_err = np.abs(exact - unmitigated)
    mit_err = np.abs(exact - mitigated)

    assert np.average(unmit_err) >= np.average(mit_err)


@pytest.mark.parametrize("n_qubits", [0, 1, 2, 3])
def test_random_projector(n_qubits):
    BASIS = np.eye(2 ** n_qubits)
    assert sample_projector(n_qubits) in BASIS


def test_random_projector_seeding():
    # test both seed types
    assert (sample_projector(6, seed=0) == sample_projector(6, seed=0)).all()
    first_sample = sample_projector(6, np.random.RandomState(123))
    second_sample = sample_projector(6, np.random.RandomState(123))
    assert (first_sample == first_sample).all
    # test that RandomState mutates as expected
    rnd_state = np.random.RandomState(123)
    different_results = False
    for _ in range(100):
        first_sample = sample_projector(6, rnd_state)
        second_sample = sample_projector(6, rnd_state)
        if not (first_sample == second_sample).all():
            different_results = True
            break
    assert different_results
