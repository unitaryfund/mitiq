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

import numpy as np

from mitiq.benchmarks.maxcut import run_maxcut

# Seed is not necessary for density matrix simulations


def test_square():
    graph = [(0, 1), (1, 2), (2, 3), (3, 0)]
    x0 = np.asarray([0.0, 0.5, 0.75, 1.0])
    out, _, _ = run_maxcut(graph, x0)
    assert np.isclose(out, -4.0)


def test_barbell():
    graph = [(0, 1), (1, 0)]
    x0 = np.asarray([0.0, 0.3])
    out, _, _ = run_maxcut(graph, x0)
    assert np.isclose(out, -2.0)


def test_noisy_square():
    graph = [(0, 1), (1, 2), (2, 3), (3, 0)]
    x0 = np.asarray([0.0, 0.5, 0.75, 1.0])
    out, _, _ = run_maxcut(graph, x0, noise=0.4)
    # When there is noise the solution should be worse.
    assert out > -4.0
