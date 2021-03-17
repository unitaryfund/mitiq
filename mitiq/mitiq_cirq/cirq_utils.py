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
"""Cirq utility functions."""

import numpy as np
import cirq
from cirq import Circuit

def execute(circ: Circuit, obs: np.ndarray) -> float:
        """Simulates noiseless wavefunction evolution and returns the
        expectation value of some observable.

        Args:
            circ: The input Cirq circuit.
            obs: The observable to measure as a NumPy array.

        Returns:
            The expectation value of obs as a float.
        """
        final_wvf = circ.final_state_vector()
        return np.real(final_wvf.conj().T @ obs @ final_wvf)
