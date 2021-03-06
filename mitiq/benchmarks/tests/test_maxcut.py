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

"""Unit tests for MaxCut benchmark."""
import numpy as np

from mitiq.benchmarks.maxcut import run_maxcut


def test_barbell():
    graph = [(0, 1), (1, 0)]
    opt_energy = -2.0

    energy, *_ = run_maxcut(graph, x0=[0.0, 0.3])
    assert np.isclose(energy, opt_energy)


def test_triangle():
    graph = [(0, 1), (1, 2), (2, 0)]
    opt_energy = -2.0

    x0 = [0.0, 0.5]
    energy, *_ = run_maxcut(graph, x0)
    assert np.isclose(energy, opt_energy)

    energy_with_noise, *_ = run_maxcut(graph, x0, noise=0.4)
    assert energy_with_noise > opt_energy
