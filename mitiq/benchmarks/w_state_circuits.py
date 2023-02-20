# Copyright (C) 2023 Unitary Fund
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

""" Functions for creating a W-state benchmarking circuit as defined in
:cite:`Cruz_2019_Efficient`"""

import numpy as np
import cirq



class GRotationGate(cirq.Gate):
    """Defines rotation gate G(p) with parameter p bounded between 0 amd 1.
    https://quantumai.google/cirq/build/custom_gates#with_parameters"""
    def __init__(self, p):
        super(GRotationGate, self)
        self.p = p

    def _num_qubits_(self):
        return 1

    def _unitary_(self):
        return np.array([
            [np.sqrt(self.p), -np.sqrt(1-self.p)],
            [np.sqrt(1-self.p), np.sqrt(self.p)]
        ])

    def _circuit_diagram_info_(self, args):
        return f"G({self.p})"

def _building_block_BGate():
# Build a linear complexity circuit

# Logarithmic time ocmplexity circuit