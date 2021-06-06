# Copyright (C) 2021 Unitary Fund
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

from typing import Counter, Dict, Union

import numpy as np

MeasurementResult = Union[Dict[bin, int], Counter[bin]]


def calculate_observable(
    state_or_measurements: Union[MeasurementResult, np.ndarray], observable: np.ndarray
) -> float:
    """Returns (estimate of) ⟨𝛹| O |𝛹⟩ for diagonal observable O and quantum
     state |𝛹⟩.

    Args:
        state_or_measurements: Quantum state to calculate the expectation
            value of the observable in. Can be provided as a wavefunction
            (numpy array) or as a dictionary of counts from sampling the
            wavefunction in the computational basis.
        observable: Observable as a diagonal matrix (one-dimensional numpy
            array).
    """
    nqubits = int(np.log2(len(observable)))

    if isinstance(state_or_measurements, np.ndarray):
        observable_values = [
            observable[i] * abs(np.conjugate(state_or_measurements[i]) * state_or_measurements[i])
            for i in range(2 ** nqubits)
        ]
    elif isinstance(state_or_measurements, (dict, Counter)):
        # order the counts and add zeros:
        state_or_measurements = measurements_to_probabilities(state_or_measurements, nqubits)
        values = list(state_or_measurements.values())
        observable_values = [
            (observable[i] * values[i]) for i in range(2 ** nqubits)
        ]
    else:
        raise ValueError(
            f"Provided state has type {type(state_or_measurements)} but must be a numpy "
            f"array or dictionary of counts."
        )

    return sum(np.real(observable_values))


def measurements_to_probabilities(
    counts: MeasurementResult, nqubits: int,
) -> MeasurementResult:
    """Normalizes the counts and inserts 0 where counts are missing."""
    total_counts = sum(counts.values())
    state = {bin(j): 0.0 for j in range(2 ** nqubits)}
    for key, value in counts.items():
        state[key] = value / total_counts
    return state
