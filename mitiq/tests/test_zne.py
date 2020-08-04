"""Unit tests for zero-noise extrapolation."""

import numpy as np
import pytest

import cirq

from mitiq.factories import LinearFactory, RichardsonFactory
from mitiq.folding import (
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
    "fold_method", [fold_gates_from_left, fold_gates_from_right, fold_gates_at_random]
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
    "fold_method", [fold_gates_from_left, fold_gates_from_right, fold_gates_at_random]
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


def test_execute_with_zne_bad_arguments():
    """Tests errors are raised when execute_with_zne is called with bad args.
    """
    with pytest.raises(TypeError, match="Argument `executor` must be callable"):
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
