# Copyright (C) 2022 Unitary Fund
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

import numpy as np

from mitiq._typing import MeasurementResult


def to_probability_vector(measurement: np.ndarray) -> np.ndarray:
    """Convert a raw measurement to a probability vector.

    Args:
        measurement: A single measurement.
    """
    index = int("".join([str(m) for m in measurement]), base=2)
    pv = np.zeros((2 ** measurement.shape[0]), dtype=np.uint8)
    pv[index] = 1
    return pv


def sample_probability_vector(probability_vector: np.ndarray) -> np.ndarray:
    """Sample a probability vector and returns a bitstring.

    Args:
        probability_vector: A probability vector.
    """
    result = np.random.choice(
        len(probability_vector), size=1, p=probability_vector
    )[0]
    value = [int(i) for i in bin(result)[2:]]
    return np.array(value, dtype=np.uint8)


def mitigate_measurements(
    noisy_result: MeasurementResult,
    inverse_confusion_matrix: np.ndarray,
) -> MeasurementResult:
    """Applies the inverse confusion matrix against the noisy measurement
    result and returns the adjusted measurements.

    Args:
        noisy_results: The unmitigated ``MeasurementResult``.
        inverse_confusion_matrix: The inverse confusion matrix to apply to the
            probability vector estimated with noisy measurement results.
    """
    if noisy_result.qubit_indices is None:
        raise ValueError(
            "Qubit indices are missing from the noisy measurement result."
        )

    num_qubits = len(noisy_result.qubit_indices)
    required_shape = (2**num_qubits, 2**num_qubits)
    if inverse_confusion_matrix.shape != required_shape:
        raise ValueError(
            f"Inverse confusion matrix should have shape {required_shape}, but"
            f" it has {inverse_confusion_matrix.shape} instead."
        )

    empirical_prob_dist = np.apply_along_axis(
        to_probability_vector, 1, noisy_result.asarray
    )

    adjusted_prob_dist = (inverse_confusion_matrix @ empirical_prob_dist.T).T

    adjusted_result = np.apply_along_axis(
        sample_probability_vector, 1, adjusted_prob_dist
    )

    result = MeasurementResult(adjusted_result, noisy_result.qubit_indices)
    return result
