# Copyright (C) 2020 Unitary Fund
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

"""Methods for scaling noise in circuits by adding or modifying gates."""
from mitiq.zne.scaling.folding import (
    fold_all,
    fold_gates_from_left,
    fold_gates_from_right,
    fold_gates_at_random,
    fold_global,
)
from mitiq.zne.scaling.parameter import (
    scale_parameters,
    compute_parameter_variance,
)
from mitiq.zne.scaling.identity_insertion import insert_id_layers
