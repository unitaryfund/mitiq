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

from functools import reduce
from typing import List
import numpy as np
import numpy.typing as npt


from mitiq._typing import MeasurementResult, Bitstring


def sample_probability_vector(
    probability_vector: npt.NDArray[np.float64], samples: int
) -> List[Bitstring]:
    """Generate a number of samples from a probability distribution as
    bitstrings.

    Args:
        probability_vector: A probability vector.

    Returns:
        A list of sampled bitstrings.
    """
    # sample using the probability distribution given
    num_values = len(probability_vector)
    choices = np.random.choice(num_values, size=samples, p=probability_vector)

    # convert samples to binary strings
    bit_width = int(np.log2(num_values))
    binary_repr_vec = np.vectorize(np.binary_repr)
    binary_strings = binary_repr_vec(choices, width=bit_width)

    # split the binary strings into an array of ints
    bitstrings = (
        np.apply_along_axis(
            np.fromstring, 1, binary_strings[:, None], dtype="U1"
        )
        .astype(np.uint8)
        .tolist()
    )

    return bitstrings


def bitstrings_to_probability_vector(
    bitstrings: List[Bitstring],
) -> npt.NDArray[np.float64]:
    """Converts a list of measured bitstrings to a probability vector estimated
    as the empirical frequency of each bitstring (ordered with increasing
    binary value).

    Args:
        bitstrings: All measured bitstrings.

    Returns:
        A probabiity vector corresponding to the measured bitstrings.
    """
    pv = np.zeros(2 ** len(bitstrings[0]))
    for bs in bitstrings:
        index = int("".join(map(str, bs)), base=2)
        pv[index] += 1
    pv /= len(bitstrings)

    return pv


def generate_inverse_confusion_matrix(
    num_qubits: int,
    p0: float = 0.01,
    p1: float = 0.01,
) -> npt.NDArray[np.float64]:
    """
    Generates the inverse confusion matrix assuming a single-qubit
    model for measurement errors. This is useful for applying
    the measurement error mitigation technique in ``mitiq.rem``.

    Args:
        num_qubits: The number of qubits in the system.
        p0: Probability of flipping a 0 to a 1.
        p1: Probability of flipping a 1 to a 0.

    Returns:
        The inverse confusion matrix.
    """
    # Use a smaller single qubit confusion matrix for generating
    # the larger inverse confusion matrix (by tensoring).
    # Implies that errors are uncorrelated among qubits.
    cm = np.array([[1 - p0, p1], [p0, 1 - p1]])
    inv_cm = np.linalg.pinv(cm)

    tensored_inv_cm = reduce(np.kron, [inv_cm] * num_qubits)
    return tensored_inv_cm


def generate_tensored_inverse_confusion_matrix(
    num_qubits: int, confusion_matrices: List[npt.NDArray[np.float64]]
) -> npt.NDArray[np.float64]:
    """
    Generates the inverse confusion matrix utilizing the supplied
    confusion matrices for individual or combined subsystems. This
    is useful for applying the measurement error mitigation
    technique in ``mitiq.rem``.

    Args:
        num_qubits: The number of qubits in the system.
        confusion_matrices: The confusion matrices for the individual
            or sometimes combined subsystems.

    Returns:
        The inverse confusion matrix.
    """
    inv_confusion_matrices = [np.linalg.pinv(cm) for cm in confusion_matrices]
    tensored_inv_cm = reduce(np.kron, inv_confusion_matrices)

    expected_shape = (2**num_qubits, 2**num_qubits)
    if tensored_inv_cm.shape != expected_shape:
        raise ValueError(
            f"The supplied confusion matrices don't produce the "
            f"correctly sized inverse confusion matrix: "
            f"{tensored_inv_cm.shape} should be {expected_shape}."
        )

    return tensored_inv_cm


def mitigate_measurements(
    noisy_result: MeasurementResult,
    inverse_confusion_matrix: npt.NDArray[np.float64],
) -> MeasurementResult:
    """Applies the inverse confusion matrix against the noisy measurement
    result and returns the adjusted measurements.

    Args:
        noisy_results: The unmitigated ``MeasurementResult``.
        inverse_confusion_matrix: The inverse confusion matrix to apply to the
            probability vector estimated with noisy measurement results.

    Returns:
        A mitigated MeasurementResult.
    """
    num_qubits = noisy_result.nqubits
    required_shape = (2**num_qubits, 2**num_qubits)
    if inverse_confusion_matrix.shape != required_shape:
        raise ValueError(
            f"Inverse confusion matrix should have shape {required_shape}, but"
            f" it has {inverse_confusion_matrix.shape} instead."
        )

    empirical_prob_dist = bitstrings_to_probability_vector(noisy_result.result)

    adjusted_prob_dist = (inverse_confusion_matrix @ empirical_prob_dist.T).T
    # remove negative values
    adjusted_prob_dist = adjusted_prob_dist.clip(min=0)
    # re-normalize, so values sum to 1
    adjusted_prob_dist /= np.sum(adjusted_prob_dist)

    adjusted_result = sample_probability_vector(
        adjusted_prob_dist, noisy_result.shots
    )

    result = MeasurementResult(adjusted_result, noisy_result.qubit_indices)
    return result
