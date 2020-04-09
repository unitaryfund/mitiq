# Examples for mitiq usage, also used in the Sphinx documentation builder.
#
import numpy as np
from cirq import Circuit, depolarize, LineQubit, X, DensityMatrixSimulator

SIMULATOR = DensityMatrixSimulator()

# 0.1% depolarizing noise
NOISE = 0.001


def noisy_simulation(circ: Circuit, shots=None) -> float:
    """ Simulates a circuit with depolarizing noise at level NOISE.

    Args:
        circ: The quantum program as a cirq object.
        shots: This unused parameter is needed to match mitiq's expected type
               signature for an executor function.

    Returns:
        The observable's measurements as as
        tuple (expectation value, variance).
    """
    circuit = circ.with_noise(depolarize(p=NOISE))
    rho = SIMULATOR.simulate(circuit).final_density_matrix
    # define the computational basis observable
    obs = np.diag([1, 0])
    expectation = np.real(np.trace(rho @ obs))
    return expectation
