"""Unit tests for zero-noise extrapolation."""

import numpy as np
import pytest

import cirq

from mitiq.matrices import npX, npZ
from mitiq.factories import (
    LinearFactory, RichardsonFactory, BatchedShotFactory
)
from mitiq.folding import (
    fold_gates_from_left, fold_gates_from_right, fold_gates_at_random
)
from mitiq.zne import execute_with_zne, mitigate_executor


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


# Mock executor with shots argument for unit tests
def executor_with_shots(circuit, shots):
    fake_noise = np.random.rand() / np.sqrt(shots)
    return executor(circuit) + fake_noise


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
    """Tests errors are raised when execute_with_zne is called with bad args.
    """
    with pytest.raises(TypeError,
                       match="Argument `executor` must be callable"):
        execute_with_zne(circ, None)

    with pytest.raises(TypeError, match="Argument `factory` must be of type"):
        execute_with_zne(circ, executor, factory=RichardsonFactory)

    with pytest.raises(TypeError, match="Argument `scale_noise` must be"):
        execute_with_zne(circ, executor, scale_noise=None)


def test_run_factory_with_number_of_shots():
    """Tests qrun of a RichardsonFactory merged with BatchedShotFactory."""

    class RichardsonShotFactory(BatchedShotFactory, RichardsonFactory):
        """Richardson extrapolation factory with shot_list argument."""
        pass

    fac = RichardsonShotFactory([1.0, 2.0, 3.0],
                                shot_list=[10 ** 6, 10 ** 6, 10 ** 6])
    fac.run(circ, executor_with_shots, scale_noise=fold_gates_from_left)
    result = fac.reduce()
    assert np.isclose(result, 0.0, atol=1.0e-2)


def test_mitigate_executor_with_shot_list():
    """Tests the mitigation of an executor using different shots
    for each noise scale factor.
    """

    class LinearShotFactory(BatchedShotFactory, LinearFactory):
        """Linear extrapolation factory with shot_list argument."""
        pass

    bad_fac = LinearShotFactory([1.0, 2.0, 3.0], shot_list=[1, 1, 1])
    mitigated_executor = mitigate_executor(executor_with_shots,
                                           factory=bad_fac)
    assert not np.isclose(mitigated_executor(circ), 0.0, atol=1.0e-3)
    good_fac = LinearShotFactory([1.0, 2.0, 3.0],
                                 shot_list=[10**9, 10**9, 10**9])
    mitigated_executor = mitigate_executor(executor_with_shots,
                                           factory=good_fac)
    assert np.isclose(mitigated_executor(circ), 0.0, atol=1.0e-3)
