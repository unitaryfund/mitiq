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

import pyquil

from mitiq import QPROGRAM
from mitiq.zne import (
    inference,
    scaling,
    execute_with_zne,
    mitigate_executor,
    zne_decorator,
)
from mitiq.mitiq_pyquil.pyquil_utils import (
    generate_qcs_executor,
    ground_state_expectation,
)
from mitiq.mitiq_pyquil.conversions import to_pyquil
from mitiq.benchmarks.randomized_benchmarking import generate_rb_circuits

TEST_DEPTH = 30


def random_one_qubit_identity_circuit(num_cliffords: int) -> pyquil.Program:
    """Returns a single-qubit identity circuit.

    Args:
        num_cliffords (int): Number of cliffords used to generate the circuit.

    Returns:
        circuit: Quantum circuit as a :class:`pyquil.Program` object.
    """
    return to_pyquil(
        *generate_rb_circuits(
            n_qubits=1, num_cliffords=num_cliffords, trials=1
        )
    )


QVM = pyquil.get_qc("1q-qvm")
QVM.qam.random_seed = 1337
noiseless_executor = generate_qcs_executor(
    qc=pyquil.get_qc("1q-qvm"),
    expectation_fn=ground_state_expectation,
    shots=1_000,
)


def test_run_factory():
    qp = random_one_qubit_identity_circuit(num_cliffords=TEST_DEPTH)

    fac = inference.RichardsonFactory([1.0, 2.0, 3.0])

    fac.run(qp, noiseless_executor, scale_noise=scaling.fold_gates_at_random)
    result = fac.reduce()
    assert np.isclose(result, 1.0, atol=1e-5)


def test_execute_with_zne():
    qp = random_one_qubit_identity_circuit(num_cliffords=TEST_DEPTH)
    result = execute_with_zne(qp, noiseless_executor)
    assert np.isclose(result, 1.0, atol=1e-5)


def test_mitigate_executor():
    qp = random_one_qubit_identity_circuit(num_cliffords=TEST_DEPTH)

    new_executor = mitigate_executor(noiseless_executor)
    result = new_executor(qp)
    assert np.isclose(result, 1.0, atol=1e-5)


@zne_decorator(scale_noise=scaling.fold_gates_at_random)
def decorated_executor(qp: QPROGRAM) -> float:
    return noiseless_executor(qp)


def test_zne_decorator():
    qp = random_one_qubit_identity_circuit(num_cliffords=TEST_DEPTH)

    result = decorated_executor(qp)
    assert np.isclose(result, 1.0, atol=1e-5)
