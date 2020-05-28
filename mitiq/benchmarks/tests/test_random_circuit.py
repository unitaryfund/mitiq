from itertools import product
import pytest

import numpy as np

from mitiq.factories import LinearFactory, RichardsonFactory, PolyFactory
from mitiq.folding import fold_gates_at_random, fold_gates_from_left, \
    fold_gates_from_right
from mitiq.benchmarks.random_circuits import rand_circuit_zne, sample_projector

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
    exact, unmitigated, mitigated = rand_circuit_zne(n_qubits=2,
                                                     depth=20,
                                                     trials=8,
                                                     op_density=0.99,
                                                     noise=0.04,
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
    # test seeding with an integer
    first_sample = sample_projector(6, seed=SEED)
    second_sample = sample_projector(6, seed=SEED)
    assert (first_sample == second_sample).all()

    # test seeding with a random state
    first_sample = sample_projector(6, np.random.RandomState(SEED))
    second_sample = sample_projector(6, np.random.RandomState(SEED))
    assert (first_sample == second_sample).all()

    # test that different random states return different projectors
    first_sample = sample_projector(6, np.random.RandomState(1))
    second_sample = sample_projector(6, np.random.RandomState(2))
    assert not (first_sample == second_sample).all()
