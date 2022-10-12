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
"""Functions for scaling supported circuits by inserting layers of identity
gates."""

import numpy as np
import random
from typing import Tuple
from cirq import Circuit, ops, Moment
from mitiq.utils import (
    _append_measurements,
    _pop_measurements,
)


class UnscalableCircuitError(Exception):
    pass


# Helper functions for identity scaling
def _check_scalable(input_circuit: Circuit) -> None:
    """Raises an error if the input circuit cannot be scaled by inserting
    identity layers.

    Args:
        circuit: Checks whether this circuit can be scaled.

    Raises:
        UnscalableCircuitError:
            * If the circuit has non-terminal measurements.
    """
    if not input_circuit.are_all_measurements_terminal():
        raise UnscalableCircuitError(
            "Circuit contains intermediate measurements and cannot be folded."
        )


def _calculate_id_layers(
    input_circuit_depth: int, scale_factor: float
) -> Tuple[int, int]:
    """Returns a tuple of integers that describes the number of identity layers
    to be inserted after each layer of the input circuit.

    Args:
        input_circuit_depth : Number of moments in the input_circuit
        scale_factor : Intended noise scaling factor

    Returns:
        (num_uniform_layers, num_partial_layers) : A tuple of the number of
        uniform identity layers to be inserted after each moment in the
        input_circuit and a number of partial layers to be inserted after
        some random moments to be able to achieve the intended scale factor.
    """
    if scale_factor < 1:
        raise ValueError(
            f"Requires scale_factor >= 1 but scale_factor = {scale_factor}."
        )

    num_uniform_layers = int(scale_factor - 1)
    int_scale_factor = num_uniform_layers + 1
    if np.isclose(int_scale_factor, scale_factor):
        return (num_uniform_layers, 0)
    else:
        num_partial_layers = int(
            input_circuit_depth * (scale_factor - 1 - num_uniform_layers)
        )
        return (num_uniform_layers, num_partial_layers)


def insert_id_layers(input_circuit: Circuit, scale_factor: float) -> Circuit:
    """Returns a scaled version of the input circuit by inserting layers of
    identities.

    Args:
        input_circuit : Cirq Circuit to be scaled
        scale_factor : Noise scaling factor as a float

    Returns:
        scaled_circuit : Scaled quantum circuit via identity layer insertions
    """
    _check_scalable(input_circuit)
    measurements = _pop_measurements(input_circuit)
    input_circuit_depth = len(input_circuit)

    num_uniform_layers, num_partial_layers = _calculate_id_layers(
        input_circuit_depth, scale_factor
    )

    random_moment_indices = random.sample(
        range(input_circuit_depth), num_partial_layers
    )

    circuit_qubits = input_circuit.all_qubits()
    id_layer = Moment(ops.I.on_each(*circuit_qubits))

    scaled_circuit = Circuit()
    for i, moment in enumerate(input_circuit):
        scaled_circuit.append(moment)

        scaled_circuit.append([id_layer] * (num_uniform_layers))

        if i in random_moment_indices:
            scaled_circuit.append(id_layer)

    _append_measurements(scaled_circuit, measurements)
    return scaled_circuit
