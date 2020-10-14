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

"""High-level probabilistic error cancellation tools."""

from typing import Optional, Callable
import numpy as np
from mitiq._typing import QPROGRAM
from mitiq.pec.utils import DecoType
from mitiq.pec.sampling import sample_circuit


def execute_with_pec(
    circuit: QPROGRAM,
    executor: Callable[[QPROGRAM], float],
    deco_dict: DecoType,
    num_to_average: int = 1,
    num_samples: Optional[int] = None,
) -> float:
    """Evaluates the expectation value associated to the input circuit
    using probabilistic error cancellation (PEC) [Temme2017]_.

    This function implements PEC by:

    1. Sampling different implementable circuits from the quasi-probability
       representation of the input circuit;
    2. Evaluating the noisy expectation values associated to the sampled
       circuits (through the "executor" function provided by the user);
    3. Estimating the ideal expectation value from a suitable linear
       combination of the noisy ones.

    Args:
        circuit = The input circuit to execute with error-mitigation.
        executor = A function which executes a circuits and returns an
            expectation value.
        deco_dict = The decomposition dictionary containing the quasi-
            probability representation of the ideal operations (those
            which are part of the input circuit).
        num_to_average: The number times the expectation value of each sampled
            circuit is evaluated, then averaged. Setting this number to values
            larger than 1 (default), reduces the statistical error associated
            to each noisy expectation value.
        num_samples: The number of noisy circuits to be sampled for PEC.
            If None, it is deduced from the amount of "negativity" of the
            quasi-probability representation of the input circuit.

    Returns:
        The PEC estimate of the ideal expectation value associated
        to the input circuit.

    .. [Temme2017] : Kristan Temme, Sergey Bravyi, Jay M. Gambetta,
        "Error mitigation for short-depth quantum circuits,"
        *Phys. Rev. Lett.* **119**, 180509 (2017),
        (https://arxiv.org/abs/1612.02058).

    .. [Endo2018] : Suguru Endo, Simon C. Benjamin, Ying Li,
        "Practical Quantum Error Mitigation for Near-Future Applications"
        *Phys. Rev. **X 8**, 031027 (2018),
        (https://arxiv.org/abs/1712.09271).

    .. [Takagi2020] : Ryuji Takagi,
        "Optimal resource cost for error mitigation,"
        (https://arxiv.org/abs/2006.12509).
    """

    # TODO: deduce num_to_sample from the "negativity" of the decomposition
    if not num_samples:
        num_samples = 1000

    sampled_circuits = []
    signs = []
    norms = []

    for i in range(num_samples):
        sampled_circuit, sign, norm = sample_circuit(circuit, deco_dict)
        sampled_circuits.append(sampled_circuit)
        signs.append(sign)
        norms.append(norm)

    # The norm of the quasi-distribution should be independent of sample
    assert np.all(norms == norms[0])
    norm = norms[0]

    # Repeat each sampled circuit "num_to_average" times
    to_run = [circ for circ in sampled_circuits for _ in range(num_to_average)]

    # TODO: deal with batched executors
    # Execute all the circuits
    results = np.array([executor(circ) for circ in to_run])

    # Reshape results to "num_to_average" columns
    results = results.reshape(-1, num_to_average)

    # Average the "num_to_average" columns
    exp_values = np.average(results, axis=1)

    # Evaluate unbiased estimators [Temme2017], [Endo2018], [Takagi2020]
    unbiased_estimators = [s * norm * val for val, s in zip(exp_values, signs)]

    # Average to return the PEC estimate of the ideal expectation value
    return np.average(unbiased_estimators)
