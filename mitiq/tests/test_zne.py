import numpy as np

# Error mitigation package
from mitiq import class_mitigator
from mitiq.tests.pyquil_utils import random_identity_circuit, run_with_noise, measure
import mitiq.tests.qiskit_utils as qiskit_utils


@class_mitigator(order=2)
def magic_run_pyquil(circuit=None, stretch=1):
    """Execute a circuit on a noisy device and returns the expectation value of
    the final measurement."""
    _circuit = circuit.copy()
    true_noise = 0.007  # real value of the noise
    noise = true_noise * stretch
    expval = run_with_noise(_circuit, noise, shots=10**3)
    return expval


@class_mitigator(order=2)
def magic_run_qiskit(circuit=None, stretch=1):
    """Execute a circuit on a noisy device and returns the expectation value of
    the final measurement."""
    true_noise = 0.007  # real value of the noise
    noise = true_noise * stretch
    expval = qiskit_utils.run_with_noise(circuit, noise, shots=10**5)
    return expval


def test_rand_circ_pyquil():
    rand_circ = random_identity_circuit(depth=30)
    rand_circ = measure(rand_circ, qid=0)

    # execution with automatic error mitigation
    xx = magic_run_pyquil(rand_circ)
    assert np.isclose(xx, 1.0, atol=1.e-2)


def test_rand_circ_qiskit():
    rand_circ = qiskit_utils.random_identity_circuit(depth=30)
    rand_circ.measure(0, 0)

    # execution with automatic error mitigation
    xx = magic_run_qiskit(rand_circ)
    assert np.isclose(xx, 1.0, atol=1.e-2)
