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

"""Unit tests for PEC."""

from functools import partial
import pytest

import numpy as np
from cirq import Circuit, LineQubit, Y, Z, CNOT

from mitiq.pec.utils import _simple_pauli_deco_dict
from mitiq.pec.pec import execute_with_pec, LargeSampleWarning
from mitiq.benchmarks.utils import noisy_simulation

# The level of depolarizing noise for the simulated backend
BASE_NOISE = 0.02

# Define some decomposition dictionaries for testing
DECO_DICT = _simple_pauli_deco_dict(BASE_NOISE)
DECO_DICT_SIMP = _simple_pauli_deco_dict(BASE_NOISE, simplify_paulis=True)
NOISELESS_DECO_DICT = _simple_pauli_deco_dict(0)


def serial_executor(circuit: Circuit, noise: float = BASE_NOISE) -> float:
    """A one- or two-qubit noisy executor function which executes the input
    circuit with `noise` depolarizing noise and returns the expectation value
    of the ground state projector.
    """
    if len(circuit.all_qubits()) == 1:
        obs = np.array([[1, 0], [0, 0]])
    elif len(circuit.all_qubits()) == 2:
        obs = np.array(
            [[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
        )
    else:
        raise ValueError("The input must be a circuit with 1 or 2 qubits.")

    return noisy_simulation(circuit, noise, obs)


def batched_executor(circuits) -> np.ndarray:
    return np.array([serial_executor(circuit) for circuit in circuits])


def fake_executor(circuit: Circuit, random_state: np.random.RandomState):
    """A fake executor which just samples from a normal distribution."""
    return random_state.randn()


# Simple identity 1-qubit circuit for testing
q = LineQubit(1)
oneq_circ = Circuit(Z.on(q), Z.on(q))

# Simple identity 2-qubit circuit for testing
qreg = LineQubit.range(2)
twoq_circ = Circuit(Y.on(qreg[1]), CNOT.on(*qreg), Y.on(qreg[1]),)


@pytest.mark.parametrize("circuit", [oneq_circ, twoq_circ])
@pytest.mark.parametrize("executor", [serial_executor, batched_executor])
@pytest.mark.parametrize(
    "decomposition_dict", [NOISELESS_DECO_DICT, DECO_DICT_SIMP, DECO_DICT]
)
@pytest.mark.parametrize("seed", (100, 101))
def test_execute_with_pec(circuit, executor, decomposition_dict, seed):
    """Tests that execute_with_pec mitigates the error of a noisy
    expectation value.
    """
    unmitigated = serial_executor(circuit)
    mitigated = execute_with_pec(
        circuit,
        executor,
        decomposition_dict=decomposition_dict,
        force_run_all=False,
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


@pytest.mark.parametrize("circuit", [oneq_circ, twoq_circ])
@pytest.mark.parametrize("seed", (1, 2, 3))
def test_execute_with_pec_with_different_samples(circuit: Circuit, seed: int):
    """Tests that, on average, the error decreases as the number of samples is
    increased.
    """
    errors_few_samples = []
    errors_more_samples = []
    for _ in range(10):
        mitigated = execute_with_pec(
            circuit,
            serial_executor,
            decomposition_dict=DECO_DICT,
            num_samples=10,
            force_run_all=False,
            random_state=seed,
        )
        errors_few_samples.append(abs(mitigated - 1.0))
        mitigated = execute_with_pec(
            circuit,
            serial_executor,
            decomposition_dict=DECO_DICT,
            num_samples=100,
            random_state=seed,
        )
        errors_more_samples.append(abs(mitigated - 1.0))

    assert np.average(errors_more_samples) < np.average(errors_few_samples)


@pytest.mark.parametrize("num_samples", [100, 1000])
def test_execute_with_pec_error(num_samples: int):
    """Tests that the error associated to the PEC value scales as
    1/sqrt(num_samples).
    """
    _, error_pec = execute_with_pec(
        oneq_circ,
        partial(fake_executor, random_state=np.random.RandomState(0)),
        DECO_DICT,
        num_samples=num_samples,
        force_run_all=True,
        full_output=True,
    )
    # The error should scale as 1/sqrt(num_samples)
    print(error_pec * np.sqrt(num_samples))
    assert np.isclose(error_pec * np.sqrt(num_samples), 1.0, atol=0.1)


@pytest.mark.parametrize("precision", [0.1, 0.01])
def test_precision_option_in_execute_with_pec(precision: float):
    """Tests that the 'precision' argument is used to deduce num_samples."""
    # For a noiseless circuit we expect num_samples = 1/precision^2:
    _, pec_error = execute_with_pec(
        oneq_circ,
        partial(fake_executor, random_state=np.random.RandomState(0)),
        NOISELESS_DECO_DICT,
        precision=precision,
        force_run_all=True,
        full_output=True,
    )
    # The error should scale as precision
    assert np.isclose(pec_error / precision, 1.0, atol=0.1)

    # If num_samples is given, precision is ignored.
    _, pec_error = execute_with_pec(
        oneq_circ,
        partial(fake_executor, random_state=np.random.RandomState(0)),
        NOISELESS_DECO_DICT,
        precision=precision,
        num_samples=1000,
        full_output=True,
    )
    # The error should scale as 1/sqrt(num_samples)
    assert not np.isclose(pec_error / precision, 1.0, atol=0.1)
    assert np.isclose(pec_error * np.sqrt(1000), 1.0, atol=0.1)


@pytest.mark.parametrize("bad_value", (0, -1, 2))
def test_bad_precision_argument(bad_value: float):
    """Tests that if 'precision' is not within (0, 1] an error is raised."""

    with pytest.raises(ValueError, match="The value of 'precision' should"):
        execute_with_pec(
            oneq_circ, serial_executor, DECO_DICT, precision=bad_value
        )


@pytest.mark.skip(reason="Slow test.")
def test_large_sample_size_warning():
    """Tests whether a warning is raised when PEC sample size
    is greater than 10 ** 5
    """
    with pytest.warns(
        LargeSampleWarning, match=r"The number of PEC samples is very large.",
    ):
        execute_with_pec(
            oneq_circ, fake_executor, DECO_DICT, num_samples=100001

    )
