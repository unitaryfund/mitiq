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

"""Unit tests for random circuits and projectors."""
from itertools import product
import pytest

import numpy as np

from mitiq.zne.inference import (
    LinearFactory,
    RichardsonFactory,
    PolyFactory,
    ExpFactory,
    AdaExpFactory,
)
from mitiq.zne.scaling import (
    fold_gates_at_random,
    fold_gates_from_left,
    fold_gates_from_right,
    fold_global,
)
from mitiq.benchmarks.random_circuits import rand_circuit_zne, sample_projector

# Fix a seed for this test file
SEED = 808


# Make fold_gates_at_random deterministic
def fold_gates_at_random_seeded(circuit, scale_factor):
    return fold_gates_at_random(circuit, scale_factor, seed=SEED)


SCALE_FUNCTIONS = [
    fold_global,
    fold_gates_at_random_seeded,
    fold_gates_from_left,
    fold_gates_from_right,
]

FACTORIES = [
    RichardsonFactory([1.0, 1.4, 2.1]),
    LinearFactory([1.0, 1.6]),
    PolyFactory([1.0, 1.4, 2.1], order=2),
    ExpFactory([1.0, 1.4, 2.1], asymptote=0.25),
    AdaExpFactory(steps=3, scale_factor=1.5, asymptote=0.25),
]


@pytest.mark.parametrize(
    ["scale_noise", "fac"], product(SCALE_FUNCTIONS, FACTORIES)
)
def test_random_benchmarks(scale_noise, fac):
    exact, unmitigated, mitigated = rand_circuit_zne(
        n_qubits=2,
        depth=20,
        trials=8,
        op_density=0.99,
        noise=0.04,
        fac=fac,
        scale_noise=scale_noise,
        seed=SEED,
    )

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
