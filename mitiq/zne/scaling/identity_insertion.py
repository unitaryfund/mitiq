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

# add comment about what QPORGRAM does
from mitiq._typing import QPROGRAM

# add comment about what converter does
from mitiq.conversions import converter

# import function defined for unitary folding
from mitiq.zne.scaling.folding import (
    _is_measurement,
    _pop_measurements,
    _append_measurements,
    _cirq_gates_to_string_keys,
)

# Define empty class when identity scaling cannot be performed on  a gate
class UnscalableGateError(Exception):
    pass


# Define empty class when identity scaling cannot be performed on  a circuit
class UnscalableCircuitError(Exception):
    pass
