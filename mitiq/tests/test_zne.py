"""Unit tests for zero-noise extrapolation."""

import numpy as np
import pytest

import cirq

from mitiq.matrices import npX, npZ
from mitiq.factories import LinearFactory, RichardsonFactory
from mitiq.folding import (
    fold_gates_from_left, fold_gates_from_right, fold_gates_at_random
)
from mitiq.zne import execute_with_zne


# Default qubit register and circuit for unit tests
qreg = cirq.GridQubit.rect(2, 1)
circ = cirq.Circuit(
    cirq.ops.H.on_each(*qreg),
    cirq.measure_each(*qreg)
)


# Default executor for unit tests
def executor(circuit):
    wavefunction = circuit.final_wavefunction()
    return np.real(
        wavefunction.conj().T @ np.kron(npX, npZ) @ wavefunction
    )


@pytest.mark.parametrize(
    "fold_method",
    [fold_gates_from_left, fold_gates_from_right, fold_gates_at_random]
)
@pytest.mark.parametrize("factory", [LinearFactory, RichardsonFactory])
def test_execute_with_zne_no_noise(fold_method, factory):
    """Tests execute_with_zne with noiseless simulation."""
    zne_value = execute_with_zne(
        circ, executor, scale_noise=fold_method, factory=factory([1., 2., 3.])
    )
    assert np.isclose(zne_value, 0.)


def test_execute_with_zne_bad_arguments():
    """Tests errors are raised when execute_with_zne is called with bad args."""
    with pytest.raises(TypeError, match="Argument `executor` must be callable"):
        execute_with_zne(circ, None)

    with pytest.raises(TypeError, match="Argument `factory` must be of type"):
        execute_with_zne(circ, executor, factory=None)

    with pytest.raises(TypeError, match="Argument `scale_noise` must be"):
        execute_with_zne(circ, executor, scale_noise=None)
