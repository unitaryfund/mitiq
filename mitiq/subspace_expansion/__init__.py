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

"""Subspace Expansion method for error mitigation as introduced in:

[1] McClean, J.R., Jiang, Z., Rubin, N.C., Babbush, R. and Neven, H., 2020.
    "Decoding quantum errors with subspace expansions". 
    (https://arxiv.org/abs/1903.05786)
[2] Yoshioka, N., Hakoshima, H., Matsuzaki, Y., Tokunaga, Y., Suzuki, Y. and Endo, S., 2022.
    "Generalized quantum subspace expansion".
    (https://arxiv.org/abs/2107.02611)

"""

from mitiq.subspace_expansion.subspace_expansion import (
    execute_with_subspace_expansion,
)

from mitiq.subspace_expansion.utils import (
    convert_from_Mitiq_Observable_to_cirq_PauliSum,
    convert_from_cirq_PauliSum_to_Mitiq_Observable,
    convert_from_cirq_PauliString_to_Mitiq_PauliString,
)
