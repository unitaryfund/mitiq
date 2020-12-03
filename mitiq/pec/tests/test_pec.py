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

from pytest import mark, raises, warns
import numpy as np
from cirq import Circuit, LineQubit, Y, Z, CNOT

from mitiq.pec.utils import _simple_pauli_deco_dict, DecompositionDict
from mitiq.pec.pec import execute_with_pec, LargeSampleWarning
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

    return noisy_simulation(circuit, BASE_NOISE, obs,)


# Simple identity 1-qubit circuit for testing
q = LineQubit(1)
oneq_circ = Circuit(Z.on(q), Z.on(q))

# Simple identity 2-qubit circuit for testing
qreg = LineQubit.range(2)
twoq_circ = Circuit(Y.on(qreg[1]), CNOT.on(*qreg), Y.on(qreg[1]),)


@mark.parametrize("seed", (100, 101))
@mark.parametrize("circuit", [oneq_circ, twoq_circ])
@mark.parametrize(
    "decomposition_dict", [NOISELESS_DECO_DICT, DECO_DICT_SIMP, DECO_DICT]
)
def test_execute_with_pec(
    circuit: Circuit, decomposition_dict: DecompositionDict, seed: int
):
    """Tests that execute_with_pec mitigates the error of a noisy
    expectation value.
    """
    unmitigated = executor(circuit)
    mitigated = execute_with_pec(
        circuit,
        executor,
        decomposition_dict=decomposition_dict,
        random_state=seed,
    )
    error_unmitigated = abs(unmitigated - 1.0)
    error_mitigated = abs(mitigated - 1.0)
    # For a trivial noiseless decomposition no PEC mitigation should happen
    if decomposition_dict == NOISELESS_DECO_DICT:
        assert np.isclose(unmitigated, mitigated)
    else:
        assert error_mitigated < error_unmitigated
        assert np.isclose(mitigated, 1.0, atol=0.1)


@mark.parametrize("circuit", [oneq_circ, twoq_circ])
@mark.parametrize("seed", (1, 2, 3))
def test_execute_with_pec_with_different_samples(circuit: Circuit, seed: int):
    """Tests that, on average, the error decreases as the number of samples is
    increased.
    """
    errors_few_samples = []
    errors_more_samples = []
    for _ in range(10):
        mitigated = execute_with_pec(
            circuit,
            executor,
            decomposition_dict=DECO_DICT,
            num_samples=10,
            random_state=seed,
        )
        errors_few_samples.append(abs(mitigated - 1.0))
        mitigated = execute_with_pec(
            circuit,
            executor,
            decomposition_dict=DECO_DICT,
            num_samples=100,
            random_state=seed,
        )
        errors_more_samples.append(abs(mitigated - 1.0))

    assert np.average(errors_more_samples) < np.average(errors_few_samples)


@mark.parametrize("num_samples", [100, 1000])
def test_execute_with_pec_with_full_output(num_samples: int):
    """Tests that the error associated to the PEC value is returned if
    the option 'full_output' is set to True.
    """
    rnd_state = np.random.RandomState(0)

    def fake_exec(circuit: Circuit):
        """A fake executor which just samples from a normal distribution."""
        return rnd_state.randn()

    _, error_pec = execute_with_pec(
        oneq_circ,
        fake_exec,
        DECO_DICT,
        num_samples=num_samples,
        full_output=True,
    )
    # The error should scale as 1/sqrt(num_samples)
    assert np.isclose(error_pec * np.sqrt(num_samples), 1.0, atol=0.1)


@mark.parametrize("precision", [0.1, 0.01])
def test_precision_option_in_execute_with_pec(precision: float):
    """Tests that the 'precision' argument is used to deduce num_samples."""

    rnd_state = np.random.RandomState(0)

    def fake_exec(circuit: Circuit):
        """A fake executor which just samples from a normal distribution."""
        return rnd_state.randn()

    # For a noiseless circuit we expect num_samples = 1/precision^2:
    _, pec_error = execute_with_pec(
        oneq_circ,
        fake_exec,
        NOISELESS_DECO_DICT,
        precision=precision,
        full_output=True,
    )
    # The error should scale as precision
    assert np.isclose(pec_error / precision, 1.0, atol=0.1)

    # If num_samples is given, precision is ignored.
    _, pec_error = execute_with_pec(
        oneq_circ,
        fake_exec,
        NOISELESS_DECO_DICT,
        precision=precision,
        num_samples=1000,
        full_output=True,
    )
    # The error should scale as 1/sqrt(num_samples)
    assert not np.isclose(pec_error / precision, 1.0, atol=0.1)
    assert np.isclose(pec_error * np.sqrt(1000), 1.0, atol=0.1)


@mark.parametrize("bad_value", (0, -1, 2))
def test_bad_precision_argument(bad_value: float):
    """Tests that if 'precision' is not within (0, 1] an error is raised."""

    with raises(ValueError, match="The value of 'precision' should"):
        execute_with_pec(oneq_circ, executor, DECO_DICT, precision=bad_value)


@mark.parametrize("num_samples", [100001])
def test_large_sample_size_warning(num_samples: int):
    """Tests whether a warning is raised when PEC sample size
    is greater than 10 ** 5
    """
    rnd_state = np.random.RandomState(0)

    def fake_exec(circuit: Circuit):
        """A fake executor which just samples from a normal distribution."""
        return rnd_state.randn()
    with warns(
        LargeSampleWarning,
        match=r"The number of PEC samples is very large.",
    ):
        execute_with_pec(
         oneq_circ, fake_exec, DECO_DICT, num_samples=num_samples)
