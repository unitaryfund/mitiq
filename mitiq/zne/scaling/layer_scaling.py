# Copyright (C) 2023 Unitary Fund
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

"""Functions for layer-wise unitary folding on supported circuits."""
from typing import List
import cirq
from cirq import inverse, Moment
from mitiq.interface import noise_scaling_converter


@noise_scaling_converter
def layer_folding(
    circuit: cirq.Circuit, layers_to_fold: List[int]
) -> cirq.Circuit:
    """Applies a variable amount of folding to select layers of a circuit.

    Args:
        circuit: The input cirq circuit.
        layers_to_fold: A list with the index referring to the layer number,
                        and the element filled by an integer represents the
                        number of times the layer is folded.

    Returns:
        A cirq ``Circuit`` with layers and number of times to fold specified
        by ``layers_to_invert``.
    """
    layers = []
    for i, layer in enumerate(circuit):
        layers.append(layer)
        # Apply the requisite number of folds to each layer.
        num_fold = layers_to_fold[i]
        for _ in range(num_fold):
            layers.append(Moment(inverse(layer)))
            layers.append(Moment(layer))

    # We combine each layer into a single circuit.
    combined_circuit = cirq.Circuit()
    for layer in layers:
        combined_circuit.append(layer)
    return combined_circuit


def layer_folding_all(
    circuit: cirq.Circuit, num_folds: int = 1
) -> List[cirq.Circuit]:
    """Return a list of cirq ``Circuit`` objects where the ith element in the
    list has the i^th layer inverted.

    Args:
        circuit: The input cirq circuit.
        layers_to_fold: A list with the index referring to the layer number,
                        and the element filled by an integer represents the
                        number of times the layer is folded.

    Returns:
        A cirq ``Circuit`` with layers and number of times to fold specified by
        ``layers_to_invert``.
    """
    return [
        layer_folding(
            circuit,
            layers_to_fold=[0] * i + [num_folds] + [0] * (len(circuit) - i),
        )
        for i in range(len(circuit))
    ]
