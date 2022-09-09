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

"""Classes corresponding to different zero-noise extrapolation methods."""
from abc import ABC, abstractmethod
from typing import List

from mitiq.interface import (
    class_atomic_one_to_many_converter,
)

import cirq
from cirq.circuits import Circuit


class AbstractCircuitGenerator(ABC):
    """Abstract base class that specifies an interface for generating new circuits
    based on a starting circuit. This includes:

        * selecting which gates to swap,
        * generating circuits based on specified gate (or other) constraints,
        * validating the generated circuits.
    """

    def __init__(self) -> None:
        return

    @abstractmethod
    def _swap_operations(
        self,
        op: cirq.ops.Operation,
    ) -> cirq.ops.Operation:
        """Calls the executor function on noise-scaled quantum circuit and
        stores the results.
        """
        raise NotImplementedError

    @abstractmethod
    @class_atomic_one_to_many_converter
    def generate_circuits(
        self,
        circuit: Circuit,
        num_circuits_to_generate: int,
    ) -> List[Circuit]:
        """Returns a list of (near) Clifford circuits obtained by replacing (some)
        non-Clifford gates in the input circuit by Clifford gates.

        The way in which non-Clifford gates are selected to be replaced is
        determined by ``method_select`` and ``method_replace``.

        In the Clifford Data Regression (CDR) method
        :cite:`Czarnik_2021_Quantum`, data generated from these circuits is used
        as a training set to learn the effect of noise.

        Args:
            circuit: The starting circuit.
            num_circuits_to_generate: The number of circuits to generate.
        """
        raise NotImplementedError
