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


from mitiq.pec.representations.depolarizing import (
    represent_operation_with_global_depolarizing_noise,
    represent_operation_with_local_depolarizing_noise,
    represent_operations_in_circuit_with_global_depolarizing_noise,
    represent_operations_in_circuit_with_local_depolarizing_noise,
    global_depolarizing_kraus,
    local_depolarizing_kraus,
)

from mitiq.pec.representations.damping import (
    _represent_operation_with_amplitude_damping_noise,
    amplitude_damping_kraus,
)

from mitiq.pec.representations.optimal import (
    minimize_one_norm,
    find_optimal_representation,
)

from mitiq.pec.representations.biased_noise import (
    represent_operation_with_local_biased_noise,
)

from mitiq.pec.representations.learning import (
    depolarizing_noise_loss_function,
    biased_noise_loss_function,
    learn_depolarizing_noise_parameter,
    learn_biased_noise_parameters,
)
