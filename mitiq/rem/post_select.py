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

from mitiq._typing import MeasurementResult


def post_select(
    measurement_result: MeasurementResult,
    hamming_weight: int,
    inverted: bool = False,
) -> MeasurementResult:
    """Discards bitstrings which do not satisfy the provided Hamming weight
    (i.e., number of 1s in the bitstring) and returns this new
    ``MeasurementResult``.

    Args:
        measurement_result: List of bitstrings.
        hamming_weight: Hamming weight of each returned bitstring (if any).
        inverted: If True, 0s count towards the Hamming weight instead of 1s.
            E.g., the inverted Hamming weight of ``[1, 0, 1]`` is 1 whereas the
            "regular" Hamming weight of this bitstring is 2.

            Note: If ``inverted`` is True, the first bitstring in
            ``measurement_result`` is used to determine the new target weight,
            so this assumes all bitstrings have the same length.
    """
    if len(measurement_result) == 0:
        return []

    if inverted:
        hamming_weight = len(measurement_result[0]) - hamming_weight

    return [bits for bits in measurement_result if sum(bits) == hamming_weight]
