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

"""Functions for identity scaling on supported circuits."""
from copy import deepcopy
from typing import (
    Any,
    Dict,
    FrozenSet,
    List,
    Optional,
    Tuple,
)

import numpy as np
from cirq import Circuit, InsertStrategy, inverse, ops, has_unitary

from mitiq._typing import QPROGRAM
from mitiq.conversions import noise_scaling_converter

from mitiq.zne.scaling.folding import ( _string_keys_to_cirq_gates,
_cirq_gates_to_string_keys, _is_measurement, _pop_measurements, _append_measurements,
_squash_moments)

class UnscalableCircuitError(Exception):
    pass

def _check_scalable(circuit: Circuit) -> None:
    """Raises an error if the input circuit cannot be scaled.
    Args:
        circuit: Checks whether this circuit is able to be scaled.
    Raises:
        UnfoldableCircuitError:
            * If the circuit has intermediate measurements.
            * If the circuit has non-unitary channels which are not terminal
              measurements.
    """
    if not circuit.are_all_measurements_terminal():
        raise UnscalableCircuitError(
            "Circuit contains intermediate measurements and cannot be scaled."
        )

    if not has_unitary(circuit):
        raise UnscalableCircuitError(
            "Circuit contains non-unitary channels which are not terminal "
            "measurements and cannot be scaled."
        )
