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

"""Tools for fitting data and extrapolating to the zero-noise limit."""
from mitiq.zne.inference.fitting import mitiq_curve_fit, mitiq_polyfit
from mitiq.zne.inference.factories import (
    AdaExpFactory,
    ExpFactory,
    LinearFactory,
    PolyExpFactory,
    PolyFactory,
    RichardsonFactory,
)
