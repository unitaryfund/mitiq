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

from collections import Counter
from typing import Counter as CounterType, Dict, Union

import numpy as np

MeasurementResult = Union[Dict[int, int], CounterType[int]]


def calculate_observable(
    state_or_measurements: Union[MeasurementResult, np.ndarray],
    observable: np.ndarray,
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
            observable[i]
            * abs(
                np.conjugate(state_or_measurements[i])
                * state_or_measurements[i]
            )
            for i in range(2 ** nqubits)
        ]
    elif isinstance(state_or_measurements, (dict, Counter)):
        probs = normalize_measurements(state_or_measurements)
        observable_values = [
            observable[i] * probs.get(i, 0.0) for i in range(2 ** nqubits)
        ]
    else:
        raise ValueError(
            f"Provided state has type {type(state_or_measurements)} but must "
            f"be a numpy array, Dict[bin, int], or Counter[int]."
        )

    return sum(np.real(observable_values))


def normalize_measurements(counts: MeasurementResult) -> Dict[int, float]:
    """Normalizes the values of the MeasurementResult to get probabilities.

    Args:
        counts: Dictionary/Counter of measurements. Each key is a binary int
            and each value is an int.
    """
    total_counts = sum(counts.values())
    return {
        bitstring: count / total_counts for bitstring, count in counts.items()
    }
