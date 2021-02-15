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

"""This module contains methods for benchmarking mitiq error extrapolation
against a standard QAOA for MAXCUT.
"""
from typing import Callable, List, Optional, Tuple
import numpy as np

from cirq import Circuit, NamedQubit, X, ZZ, H, DensityMatrixSimulator
from cirq import identity_each as id

from scipy.optimize import minimize

from mitiq import execute_with_zne, QPROGRAM
from mitiq.zne.inference import Factory
from mitiq.benchmarks.utils import noisy_simulation

SIMULATOR = DensityMatrixSimulator()


def make_noisy_backend(
    noise: float, obs: np.ndarray
) -> Callable[[Circuit], float]:
    """Helper function to match mitiq's backend type signature.

    Args:
        noise: The level of depolarizing noise.
        obs: The observable that the backend should measure.

    Returns:
        A mitiq backend function.

    """

    def noisy_backend(circ: Circuit) -> float:
        return noisy_simulation(circ, noise, obs)

    return noisy_backend


def make_maxcut(
    graph: List[Tuple[int, int]],
    noise: float = 0,
    scale_noise: Optional[Callable[[QPROGRAM, float], QPROGRAM]] = None,
    factory: Optional[Factory] = None,
) -> Tuple[
    Callable[[np.ndarray], float], Callable[[np.ndarray], Circuit], np.ndarray
]:
    """Makes an executor that evaluates the QAOA ansatz at a given beta
    and gamma parameters.

    Args:
        graph: The MAXCUT graph as a list of edges with integer labelled nodes.
        noise: The level of depolarizing noise.
        scale_noise: The noise scaling method for ZNE.
        factory: The factory to use for ZNE.

    Returns:
        (ansatz_eval, ansatz_maker, cost_obs) as a triple where

        * **ansatz_eval** -- function that evalutes the maxcut ansatz on the
            noisy cirq backend.
        * **ansatz_maker** -- function that returns an ansatz circuit.
        * **cost_obs** -- the cost observable as a dense matrix.

    """
    # get the list of unique nodes from the list of edges
    nodes = list({node for edge in graph for node in edge})
    nodes = list(range(max(nodes) + 1))

    # one qubit per node
    qreg = [NamedQubit(str(nn)) for nn in nodes]

    def cost_step(beta: float) -> Circuit:
        return Circuit(ZZ(qreg[u], qreg[v]) ** (beta) for u, v in graph)

    def mix_step(gamma: float) -> Circuit:
        return Circuit(X(qq) ** gamma for qq in qreg)

    init_state_prog = Circuit(H.on_each(qreg))

    def qaoa_ansatz(params: np.ndarray) -> Circuit:
        half = int(len(params) / 2)
        betas, gammas = params[:half], params[half:]
        qaoa_steps = sum(
            [
                cost_step(beta) + mix_step(gamma)
                for beta, gamma in zip(betas, gammas)
            ],
            Circuit(),
        )
        return init_state_prog + qaoa_steps

    # make the cost observable
    identity = np.eye(2 ** len(nodes))
    cost_mat = -0.5 * sum(
        identity - Circuit([id(*qreg), ZZ(qreg[i], qreg[j])]).unitary()
        for i, j in graph
    )
    noisy_backend = make_noisy_backend(noise, cost_mat)

    # must have this function signature to work with scipy minimize
    def qaoa_cost(params: np.ndarray) -> float:
        qaoa_prog = qaoa_ansatz(params)
        if scale_noise is None and factory is None:
            return noisy_backend(qaoa_prog)
        else:
            assert scale_noise is not None
            return execute_with_zne(
                qaoa_prog,
                executor=noisy_backend,
                scale_noise=scale_noise,
                factory=factory,
            )

    return qaoa_cost, qaoa_ansatz, cost_mat


def run_maxcut(
    graph: List[Tuple[int, int]],
    x0: np.ndarray,
    noise: float = 0,
    scale_noise: Optional[Callable[[QPROGRAM, float], QPROGRAM]] = None,
    factory: Optional[Factory] = None,
    verbose: bool = True,
) -> Tuple[float, np.ndarray, List[float]]:
    """Solves MAXCUT using QAOA on a cirq wavefunction simulator using a
       Nelder-Mead optimizer.

    Args:
        graph: The MAXCUT graph as a list of edges with integer labelled nodes.
        x0: The initial parameters for QAOA [betas, gammas].
            The size of x0 determines the number of p steps.
        noise: The level of depolarizing noise.
        scale_noise: The noise scaling method for ZNE.
        factory: The factory to use for ZNE.
        verbose: An option to pass to minimize.

    Returns:
        A triple of the minimum cost, the values of beta and gamma that
        obtained that cost, and a list of costs at each iteration step.

    Example:
        Run MAXCUT with 2 steps such that betas = [1.0, 1.1] and
        gammas = [1.4, 0.7] on a graph with four edges and four nodes.

        >>> graph = [(0, 1), (1, 2), (2, 3), (3, 0)]
        >>> fun,x,traj = run_maxcut(graph, x0=[1.0, 1.1, 1.4, 0.7])
        Optimization terminated successfully.
                 Current function value: -4.000000
                 Iterations: 108
                 Function evaluations: 188

    """
    qaoa_cost, _, _ = make_maxcut(graph, noise, scale_noise, factory)

    # store the optimization trajectories
    traj = []

    def callback(xk: np.ndarray) -> bool:
        traj.append(qaoa_cost(xk))
        return True

    res = minimize(
        qaoa_cost,
        x0=x0,
        method="Nelder-Mead",
        callback=callback,
        options={"disp": verbose},
    )

    return res.fun, res.x, traj
