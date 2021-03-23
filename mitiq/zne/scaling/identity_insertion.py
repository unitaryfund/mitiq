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
    Callable,
    Collection,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
    Union,
)

import numpy as np
from cirq import Circuit, InsertStrategy, inverse, ops, has_unitary

# Import functions from different parts of return_mitiq

# typing is used to specify what should be imprted based on installed packages
# QPROGRAM is used to specify the type of input circuit - cirq, pyquil or qiskit
from mitiq._typing import QPROGRAM

# converter is a decorator used to convert a circuit to a different type
from mitiq.conversions import converter

# import functions defined for unitary folding
from mitiq.zne.scaling.folding import (
    _is_measurement,  # checks if a gate is a measurement gate
    _pop_measurements,  # a measurement gate is removed from the circuit
    _append_measurements,
    # measurement gate is moved to the end of the circuit
    _cirq_gates_to_string_keys,
    _default_weight,
    # if weight of a gate is not specified, default value specifies it
    _get_weight_for_gate,
    # checks if gate weight is specified, if not then a default value is given
    _compute_weight, # calculates weight of a circuit
)

# Define empty class when identity scaling cannot be performed on  a gate
class UnscalableGateError(Exception):
    pass


# Define empty class when identity scaling cannot be performed on  a circuit
class UnscalableCircuitError(Exception):
    pass

# Note - this function has identical code to check_foldable. Error messages
# have been edited to raise error for identity insertion.
def _check_scalable(circuit: Circuit) -> None:
    """Raises an error if identity gates cannot be inserted.

    Args:
        circuit: Checks whether this circuit is able to be scaled by identity.

    Raises:
        UnscalableCircuitError:
            * If the circuit has intermediate measurements.
            * If the circuit has non-unitary channels which are not terminal
              measurements.
    """
    if not circuit.are_all_measurements_terminal():
        raise UnscalableCircuitError(
            "Circuit contains intermediate measurements and cannot be scaled "
            "by inserting identity gates."
        )

    if not has_unitary(circuit):
        raise UnscalableCircuitError(
            "Circuit contains non-unitary channels which are not terminal "
            "measurements and cannot be scaled by inserting identity gates."
        )

# Note - this function has identical code to get_num_to_scale. Equation
# for value being returned is changed in order to account for only 1 Identity
# being inserted instead of a gate and its hermitian.
def _get_num_to_scale(scale_factor: float, ngates: int) -> int:
    """Returns the number of gates to insert to achieve the desired
    (approximate) scale factor.

    Args:
        scale_factor: Floating point value to scale the circuit by.
        ngates: Number of gates in the circuit to fold.
    """
    return int(round(ngates * (scale_factor - 1.0)))
