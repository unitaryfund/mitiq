# /benchmarks/maxcut.py
"""
This module contains methods for benchmarking mitiq error extrapolation against
a standard QAOA for MAXCUT.
"""
from typing import List, Tuple, Callable
import numpy as np

from cirq import Circuit, NamedQubit, X, ZZ, H, DensityMatrixSimulator
from cirq import identity_each as id

from scipy.optimize import minimize

from mitiq import execute_with_zne
from mitiq.factories import Factory
from mitiq.benchmarks.utils import noisy_simulation

SIMULATOR = DensityMatrixSimulator()


def make_noisy_backend(noise: float, obs: np.ndarray) \
        -> Callable[[Circuit, int], float]:
    """ Helper function to match mitiq's backend type signature.

    Args:
        noise: The level of depolarizing noise.
        obs: The observable that the backend should measure.

    Returns:
        A mitiq backend function.
    """
    def noisy_backend(circ: Circuit):
        return noisy_simulation(circ, noise, obs)

    return noisy_backend


def make_maxcut(graph: List[Tuple[int, int]],
                noise: float = 0,
                scale_noise: Callable = None,
                factory: Factory = None
                ) -> Tuple[Callable[[np.ndarray], float],
                               Callable[[np.ndarray], Circuit], np.ndarray]:
    """Makes an executor that evaluates the QAOA ansatz at a given beta
    and gamma parameters.

    Args:
        graph: The MAXCUT graph as a list of edges with integer labelled nodes.
        noise: The level of depolarizing noise.
        scale_noise: The noise scaling method for ZNE.
        factory: The factory to use for ZNE.

    Returns:
        (ansatz_eval, ansatz_maker, cost_obs) as a triple. Here
            ansatz_eval: function that evalutes the maxcut ansatz on
                the noisy cirq backend.
            ansatz_maker: function that returns an ansatz circuit.
            cost_obs: the cost observable as a dense matrix.
    """
    # get the list of unique nodes from the list of edges
    nodes = list({node for edge in graph for node in edge})

    # one qubit per node
    qreg = [NamedQubit(str(nn)) for nn in nodes]

    def cost_step(beta):
        return Circuit(ZZ(qreg[u], qreg[v]) ** (beta) for u, v in graph)

    def mix_step(gamma):
        return Circuit(X(qq) ** gamma for qq in qreg)

    init_state_prog = Circuit(H.on_each(qreg))

    def qaoa_ansatz(params):
        half = int(len(params) / 2)
        betas, gammas = params[:half], params[half:]
        qaoa_steps = sum([cost_step(beta) + mix_step(gamma)
                          for beta, gamma in zip(betas, gammas)], Circuit())
        return init_state_prog + qaoa_steps

    # make the cost observable
    identity = np.eye(2 ** len(nodes))
    cost_mat = -0.5 * sum(identity - Circuit(
                    [id(*qreg), ZZ(qreg[i], qreg[j])]).unitary()
                for i, j in graph)
    noisy_backend = make_noisy_backend(noise, cost_mat)

    # must have this function signature to work with scipy minimize
    def qaoa_cost(params):
        qaoa_prog = qaoa_ansatz(params)
        if scale_noise is None and factory is None:
            return noisy_backend(qaoa_prog)
        else:
            return execute_with_zne(qaoa_prog,
                                    executor=noisy_backend,
                                    scale_noise=scale_noise,
                                    fac=factory)

    return qaoa_cost, qaoa_ansatz, cost_mat


def run_maxcut(graph: List[Tuple[int, int]],
               x0: np.ndarray,
               noise: float = 0,
               scale_noise: Callable = None,
               factory: Factory = None
               ) -> Tuple[float, np.ndarray, List]:
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
        A triple of the minimum cost, the values of beta and gamma that
        obtained that cost, and a list of costs at each iteration step.

    Example:
        >>> graph = [(0, 1), (1, 2), (2, 3), (3, 0)]
        >>> run_maxcut(graph, x0=[1.0, 1.1, 1.4, 0.7])
        Runs MAXCUT with 2 steps such that betas = [1.0, 1.1] and
        gammas = [1.4, 0.7]
    """
    qaoa_cost, _, _ = make_maxcut(graph, noise, scale_noise, factory)

    # store the optimization trajectories
    traj = []

    def callback(xk) -> bool:
        traj.append(qaoa_cost(xk))
        return True

    res = minimize(qaoa_cost,
                   x0=x0,
                   method='Nelder-Mead',
                   callback=callback,
                   options={'disp': True})

    return res.fun, res.x, traj
