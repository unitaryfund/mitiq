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

"""Tests for zne.py with PyQuil backend."""
import numpy as np
from mitiq._typing import QPROGRAM
from mitiq.zne.inference import RichardsonFactory
from mitiq.zne import (
    execute_with_zne,
    mitigate_executor,
    zne_decorator,
)
from mitiq.mitiq_pyquil.pyquil_utils import (
    random_identity_circuit,
    measure,
    run_program,
    scale_noise,
)

TEST_DEPTH = 30


def basic_executor(qp: QPROGRAM, shots: int = 500) -> float:
    return run_program(qp, shots)


def test_run_factory():
    rand_circ = random_identity_circuit(depth=TEST_DEPTH)
    qp = measure(rand_circ, qid=0)
    fac = RichardsonFactory([1.0, 2.0, 3.0])
    fac.run(qp, basic_executor, scale_noise)
    result = fac.reduce()
    assert np.isclose(result, 1.0, atol=1.0e-1)


def test_execute_with_zne():
    rand_circ = random_identity_circuit(depth=TEST_DEPTH)
    qp = measure(rand_circ, qid=0)
    result = execute_with_zne(qp, basic_executor, None, scale_noise)
    assert np.isclose(result, 1.0, atol=1.0e-1)


def test_mitigate_executor():
    rand_circ = random_identity_circuit(depth=TEST_DEPTH)
    qp = measure(rand_circ, qid=0)
    new_executor = mitigate_executor(basic_executor, None, scale_noise)
    # bad_result is computed with native noise (scale = 1)
    bad_result = basic_executor(scale_noise(qp, 1))
    good_result = new_executor(qp)
    assert not np.isclose(bad_result, 1.0, atol=1.0e-1)
    assert np.isclose(good_result, 1.0, atol=1.0e-1)


@zne_decorator(None, scale_noise)
def decorated_executor(qp: QPROGRAM) -> float:
    return basic_executor(qp)


def test_zne_decorator():
    rand_circ = random_identity_circuit(depth=TEST_DEPTH)
    qp = measure(rand_circ, qid=0)
    # bad_result is computed with native noise (scale = 1)
    bad_result = basic_executor(scale_noise(qp, 1))
    good_result = decorated_executor(qp)
    assert not np.isclose(bad_result, 1.0, atol=1.0e-1)
    assert np.isclose(good_result, 1.0, atol=1.0e-1)
