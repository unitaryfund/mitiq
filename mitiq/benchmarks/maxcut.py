# /benchmarks/maxcut.py
"""
This module contains methods for benchmarking mitiq error extrapolation against
a standard QAOA for MAXCUT.
"""
from typing import List, Tuple, Callable
import numpy as np

from cirq import Circuit, NamedQubit, X, ZZ, H, \
    DensityMatrixSimulator, depolarize

from pyquil.paulis import sZ, sI
from pyquil.simulation.tools import lifted_pauli

from scipy.optimize import minimize

from mitiq import execute_with_zne
from mitiq.factories import Factory

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
    # define the computational basis observable
    expectation = np.real(np.trace(rho @ obs))
    return expectation


def make_noisy_backend(noise: float, obs: np.ndarray) \
        -> Callable[[Circuit, int], float]:
    """ Helper function to match mitiq's backend type signature.

    Args:
        noise: The level of depolarizing noise.
        obs: The observable that the backend should measure.

    Returns:
        A mitiq backend function.
    """
    # shots is none to set the required type correctly.
    def noisy_backend(circ:Circuit, shots=None):
        return noisy_simulation(circ, noise, obs)
    return noisy_backend


def run_maxcut(graph: List[Tuple[int, int]],
               x0: np.ndarray,
               noise: float=0,
               scale_noise: Callable=None,
               factory: Factory=None
    ) -> Tuple[float, np.ndarray]:
    """Solves MAXCUT using QAOA on a cirq wavefunction simulator using a
       Nelder-Mead optimizer.

    Args:
        graph: The MAXCUT graph as a list of edges with integer labelled nodes.
        x0: The initial parameters for QAOA [betas, gammas].
            The size of x0 determines the number of p steps.
        noise: The level of depolarizing noise.
        scale_noise: The noise scaling method for ZNE.
        factory: The factory to use for ZNE.

    Returns:
        A tuple of the minimum cost and the values of beta and gamma that
        obtained that cost.

    Example:
        >>> graph = [(0, 1), (1, 2), (2, 3), (3, 0)]
        >>> run_maxcut(graph, x0=[1.0, 1.1, 1.4, 0.7])
        Runs MAXCUT with 2 steps such that betas = [1.0, 1.1] and
        gammas = [1.4, 0.7]
    """
    # get the list of unique nodes from the list of edges
    nodes = list({node for edge in graph for node in edge})

    # one qubit per node
    qreg = [NamedQubit(str(nn)) for nn in nodes]

    def cost_step(alpha):
        return Circuit(ZZ(qreg[u], qreg[v]) ** (alpha) for u, v in graph)

    def mix_step(beta):
        return Circuit(X(qq) ** beta for qq in qreg)

    def qaoa_ansatz(betas, gammas):
        assert len(betas) == len(gammas), "Must have the same number of " \
                                          "beta and gamma parameters."
        return sum([cost_step(beta) + mix_step(gamma)
                    for beta, gamma in zip(betas, gammas)], Circuit())

    # use pyQuil paulis as shorthand to make the dense cost operator
    h_cost = -0.5 * sum(sI(0) - sZ(i) * sZ(j) for i, j in graph)
    cost_mat = lifted_pauli(h_cost, nodes)
    noisy_backend = make_noisy_backend(noise, cost_mat)

    init_state_prog = Circuit(H.on_each(qreg))

    # must have this function signature to work with scipy minimize
    def qaoa_cost(params):
        half = int(len(params)/2)
        betas, gammas = params[:half], params[half:]
        qaoa_prog = init_state_prog + qaoa_ansatz(betas, gammas)
        return noisy_backend(qaoa_prog)

    res = minimize(qaoa_cost,
                   x0=x0,
                   method='Nelder-Mead',
                   options={'disp': True})

    return res.fun, res.x
