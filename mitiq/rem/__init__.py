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

"""Readout error mitigation (REM) techniques."""

from mitiq.rem.post_select import post_select
from mitiq.rem.inverse_confusion_matrix import (
    sample_probability_vector,
    bitstrings_to_probability_vector,
    generate_inverse_confusion_matrix,
    generate_tensored_inverse_confusion_matrix,
    mitigate_measurements,
)
from mitiq.rem.rem import (
    execute_with_rem,
    mitigate_executor,
    rem_decorator,
    mitigate_measurements,
)
