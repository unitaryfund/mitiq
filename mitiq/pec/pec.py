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

from typing import Optional, Callable, Union, Tuple
import numpy as np
import warnings
from mitiq._typing import QPROGRAM
from mitiq.pec.utils import DecompositionDict
from mitiq.pec.sampling import sample_circuit


class LargeSampleWarning(Warning):
    """Warning is raised when PEC sample size is greater than 10 ** 5
    """

    pass


_LARGE_SAMPLE_WARN = (
    "The number of PEC samples is very large. It may take several minutes."
    " It may be necessary to reduce 'precision' or 'num_samples'."
)


def execute_with_pec(
    circuit: QPROGRAM,
    executor: Callable[[QPROGRAM], float],
    decomposition_dict: DecompositionDict,
    precision: float = 0.03,
    num_samples: Optional[int] = None,
    random_state: Optional[Union[int, np.random.RandomState]] = None,
    full_output: bool = False,
) -> Union[float, Tuple[float, float]]:
    """Evaluates the expectation value associated to the input circuit
    using probabilistic error cancellation (PEC) [Temme2017]_ [Endo2018]_.

    This function implements PEC by:

    1. Sampling different implementable circuits from the quasi-probability
       representation of the input circuit;
    2. Evaluating the noisy expectation values associated to the sampled
       circuits (through the "executor" function provided by the user);
    3. Estimating the ideal expectation value from a suitable linear
       combination of the noisy ones.

    Args:
        circuit: The input circuit to execute with error-mitigation.
        executor: A function which executes a circuit and returns an
            expectation value.
        decomposition_dict: The decomposition dictionary containing the
            quasi-probability representation of the ideal operations (those
            which are part of the input circuit).
        num_samples: The number of noisy circuits to be sampled for PEC.
            If not given, this is deduced from the argument 'precision'.
        precision: The desired estimation precision (assuming the observable
            is bounded by 1). The number of samples is deduced according
            to the formula (one_norm / precision) ** 2, where 'one_norm'
            is related to the negativity of the quasi-probability
            representation [Temme2017]_. If 'num_samples' is explicitly set
            by the user, 'precision' is ignored and has no effect.
        random_state: Seed for sampling circuits.
        full_output: If False only the average PEC value is returned.
            If True an estimate of the associated error is returned too.

    Returns:
        pec_value: The PEC estimate of the ideal expectation value associated
            to the input circuit.
        pec_error: The estimated error between the mitigated 'pec_value' and
            the actual ideal expectation value. This is estimated as the ratio
            pec_std / sqrt(num_samples), where 'pec_std' is the
            standard deviation of the PEC samples, i.e., the square root of
            the mean squared deviation of the sampled values from 'pec_value'.
            This is returned only if 'full_output' is True.

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
    if isinstance(random_state, int):
        random_state = np.random.RandomState(random_state)

    # Get the 1-norm of the circuit quasi-probability representation
    _, _, norm = sample_circuit(circuit, decomposition_dict)

    if not (0 < precision <= 1):
        raise ValueError(
            "The value of 'precision' should be within the interval (0, 1],"
            f" but precision is {precision}."
        )

    # Deduce the number of samples (if not given by the user)
    if not isinstance(num_samples, int):
        num_samples = int((norm / precision) ** 2)

    # Issue warning for very large sample size
    if num_samples > 10 ** 5:
        warnings.warn(_LARGE_SAMPLE_WARN, LargeSampleWarning)

    sampled_circuits = []
    signs = []

    for _ in range(num_samples):
        sampled_circuit, sign, _ = sample_circuit(
            circuit, decomposition_dict, random_state
        )
        sampled_circuits.append(sampled_circuit)
        signs.append(sign)

    # TODO gh-412: Add support for batched executors in the PEC module
    # Execute all the circuits
    exp_values = [executor(circ) for circ in sampled_circuits]

    # Evaluate unbiased estimators [Temme2017] [Endo2018] [Takagi2020]
    unbiased_estimators = [norm * s * val for s, val in zip(signs, exp_values)]

    pec_value = np.average(unbiased_estimators)

    if full_output:
        pec_error = np.std(unbiased_estimators) / np.sqrt(num_samples)
        return pec_value, pec_error

    return pec_value
