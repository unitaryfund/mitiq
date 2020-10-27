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

"""Tests related to mitiq.pec.pec functions."""

from pytest import mark
import numpy as np
from cirq import Circuit, LineQubit, Y, Z, CNOT

from mitiq.pec.utils import _simple_pauli_deco_dict, DecoType
from mitiq.pec.pec import execute_with_pec
from mitiq.benchmarks.utils import noisy_simulation

# The level of depolarizing noise for the simulated backend
BASE_NOISE = 0.02
# Define some decomposition dictionaries for testing
DECO_DICT = _simple_pauli_deco_dict(BASE_NOISE)
DECO_DICT_SIMP = _simple_pauli_deco_dict(BASE_NOISE, simplify_paulis=True)
NOISELESS_DECO_DICT = _simple_pauli_deco_dict(0)


def executor(circuit: Circuit) -> float:
    """A one- or two-qubit noisy executor function.
    It executes the input circuit with BASE_NOISE depolarizing noise and
    returns the expectation value of the ground state projector.
    """
    if len(circuit.all_qubits()) == 1:
        obs = np.array([[1, 0], [0, 0]])
    elif len(circuit.all_qubits()) == 2:
        obs = np.array(
            [[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
        )
    else:
        raise ValueError("The input must be a circuit with 1 or 2 qubits.")

    return noisy_simulation(
        circuit,
        BASE_NOISE,
        obs,
    )


# Simple identity 1-qubit circuit for testing
q = LineQubit(1)
oneq_circ = Circuit(Z.on(q), Z.on(q))

# Simple identity 2-qubit circuit for testing
qreg = LineQubit.range(2)
twoq_circ = Circuit(
    Y.on(qreg[1]),
    CNOT.on(*qreg),
    Y.on(qreg[1]),
)


@mark.parametrize("circuit", [oneq_circ, twoq_circ])
@mark.parametrize(
    "deco_dict", [NOISELESS_DECO_DICT, DECO_DICT_SIMP, DECO_DICT]
)
def test_execute_with_pec_one_qubit(circuit: Circuit, deco_dict: DecoType):
    """Tests that execute_with_pec mitigates the error of a noisy
    expectation value.
    """
    unmitigated = executor(circuit)
    mitigated = execute_with_pec(circuit, executor, deco_dict=deco_dict)
    error_unmitigated = abs(unmitigated - 1.0)
    error_mitigated = abs(mitigated - 1.0)
    # For a trivial noiseless decomposition no PEC mitigation should happen
    if deco_dict == NOISELESS_DECO_DICT:
        assert np.isclose(unmitigated, mitigated)
    else:
        assert error_mitigated < error_unmitigated
        assert np.isclose(mitigated, 1.0, atol=0.1)


@mark.parametrize("circuit", [oneq_circ, twoq_circ])
def test_execute_with_pec_with_different_samples(circuit: Circuit):
    """Tests that, on average, the error decreases as the number of samples is
    increased.
    """
    errors_few_samples = []
    errors_more_samples = []
    for _ in range(10):
        mitigated = execute_with_pec(
            circuit, executor, deco_dict=DECO_DICT, num_samples=10
        )
        errors_few_samples.append(abs(mitigated - 1.0))
        mitigated = execute_with_pec(
            circuit, executor, deco_dict=DECO_DICT, num_samples=100
        )
        errors_more_samples.append(abs(mitigated - 1.0))

    assert np.average(errors_more_samples) < np.average(errors_few_samples)
