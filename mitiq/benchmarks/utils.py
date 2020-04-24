import numpy as np

from cirq import Circuit, depolarize, DensityMatrixSimulator

SIMULATOR = DensityMatrixSimulator()


def noisy_simulation(circ: Circuit, noise: float, obs: np.ndarray) -> float:
    """ Simulates a circuit with depolarizing noise at level NOISE.

    Args:
        circ: The quantum program as a cirq object.
        noise: The level of depolarizing noise.
        obs: The observable that the backend should measure.

    Returns:
        The observable's expectation value.
    """
    circuit = circ.with_noise(depolarize(p=noise))
    rho = SIMULATOR.simulate(circuit).final_density_matrix
    # measure the expectation by taking the trace of the density matrix
    expectation = np.real(np.trace(rho @ obs))
    return expectation
