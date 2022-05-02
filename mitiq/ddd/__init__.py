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

"""Digital dynamical decoupling (DDD) module."""

from mitiq.ddd import rules

from mitiq.ddd import insertion

from mitiq.ddd.insertion import (
    get_slack_matrix_from_circuit_mask,
    insert_ddd_sequences,
)

from mitiq.ddd.ddd import execute_with_ddd, mitigate_executor, ddd_decorator
