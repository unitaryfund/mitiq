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

"""Unit tests for zero-noise extrapolation."""
import numpy as np
import pytest

import cirq

from mitiq.zne.inference import LinearFactory, RichardsonFactory
from mitiq.zne.scaling import (
    fold_gates_from_left,
    fold_gates_from_right,
    fold_gates_at_random,
)
from mitiq.zne import execute_with_zne, mitigate_executor, zne_decorator

npX = np.array([[0, 1], [1, 0]])
"""Defines the sigma_x Pauli matrix in SU(2) algebra as a (2,2) `np.array`."""

npZ = np.array([[1, 0], [0, -1]])
"""Defines the sigma_z Pauli matrix in SU(2) algebra as a (2,2) `np.array`."""

# Default qubit register and circuit for unit tests
qreg = cirq.GridQubit.rect(2, 1)
circ = cirq.Circuit(cirq.ops.H.on_each(*qreg), cirq.measure_each(*qreg))


# Default executor for unit tests
def executor(circuit) -> float:
    wavefunction = circuit.final_wavefunction()
    return np.real(wavefunction.conj().T @ np.kron(npX, npZ) @ wavefunction)


@pytest.mark.parametrize(
    "fold_method",
    [fold_gates_from_left, fold_gates_from_right, fold_gates_at_random],
)
@pytest.mark.parametrize("factory", [LinearFactory, RichardsonFactory])
@pytest.mark.parametrize("num_to_average", [1, 2, 5])
def test_execute_with_zne_no_noise(fold_method, factory, num_to_average):
    """Tests execute_with_zne with noiseless simulation."""
    zne_value = execute_with_zne(
        circ,
        executor,
        num_to_average=num_to_average,
        scale_noise=fold_method,
        factory=factory([1.0, 2.0, 3.0]),
    )
    assert np.isclose(zne_value, 0.0)


@pytest.mark.parametrize("factory", [LinearFactory, RichardsonFactory])
@pytest.mark.parametrize(
    "fold_method",
    [fold_gates_from_left, fold_gates_from_right, fold_gates_at_random],
)
def test_averaging_improves_zne_value_with_fake_noise(factory, fold_method):
    """Tests that averaging with Gaussian noise produces a better ZNE value
    compared to not averaging with several folding methods.

    For non-deterministic folding, the ZNE value with average should be better.
    For deterministic folding, the ZNE value should be the same.
    """
    for seed in range(5):
        rng = np.random.RandomState(seed)

        def noisy_executor(circuit) -> float:
            return executor(circuit) + rng.randn()

        zne_value_no_averaging = execute_with_zne(
            circ,
            noisy_executor,
            num_to_average=1,
            scale_noise=fold_gates_at_random,
            factory=factory([1.0, 2.0, 3.0]),
        )

        zne_value_averaging = execute_with_zne(
            circ,
            noisy_executor,
            num_to_average=10,
            scale_noise=fold_method,
            factory=factory([1.0, 2.0, 3.0]),
        )

        # True (noiseless) value is zero. Averaging should ==> closer to zero.
        assert abs(zne_value_averaging) <= abs(zne_value_no_averaging)


@pytest.mark.parametrize("factory", [LinearFactory, RichardsonFactory])
def test_execute_with_zne_fit_fail(factory):
    """Tests errors are raised when asking for fitting parameters that can't
    be calculated.
    """
    with pytest.raises(
        ValueError,
        match="""Data is either ill-defined or not enough to evaluate the
        required information. Please make sure that the 'run' and 'reduce'
        methods have been called and that enough expectation values have been
        measured.""",
    ):
        factory([1.0, 2.0]).get_zero_noise_limit_error()
        factory([1.0, 2.0]).get_optimal_parameters()
        factory([1.0, 2.0]).get_parameters_covariance()
        factory([1.0, 2.0]).get_zero_noise_limit()
        factory([1.0, 2.0]).get_extrapolation_curve()


def test_execute_with_zne_bad_arguments():
    """Tests errors are raised when execute_with_zne is called with bad args.
    """
    with pytest.raises(
        TypeError, match="Argument `executor` must be callable"
    ):
        execute_with_zne(circ, None)

    with pytest.raises(TypeError, match="Argument `factory` must be of type"):
        execute_with_zne(circ, executor, factory=RichardsonFactory)

    with pytest.raises(TypeError, match="Argument `scale_noise` must be"):
        execute_with_zne(circ, executor, scale_noise=None)


def test_error_zne_decorator():
    """Tests that the proper error is raised if the decorator is
    used without parenthesis.
    """
    with pytest.raises(TypeError, match="Decorator must be used with paren"):

        @zne_decorator
        def test_executor(circuit):
            return 0


def test_doc_is_preserved():
    """Tests that the doc of the original executor is preserved."""

    def first_executor(circuit):
        """Doc of the original executor."""
        return 0

    mit_executor = mitigate_executor(first_executor)
    assert mit_executor.__doc__ == first_executor.__doc__

    @zne_decorator()
    def second_executor(circuit):
        """Doc of the original executor."""
        return 0

    assert second_executor.__doc__ == first_executor.__doc__
