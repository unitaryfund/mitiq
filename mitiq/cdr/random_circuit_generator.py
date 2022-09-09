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
from typing import (
    cast,
    List,
    Optional,
    Sequence,
    Union,
)

import numpy as np
import cirq
from cirq.circuits import Circuit

from mitiq.cdr.circuit_generator import AbstractCircuitGenerator
from mitiq.cdr.clifford_utils import (
    random_clifford,
    closest_clifford,
    angle_to_proximity,
    probabilistic_angle_to_clifford,
)

from mitiq.interface import (
    class_atomic_one_to_many_converter,
)

_GAUSSIAN = "gaussian"
_UNIFORM = "uniform"
_CLOSEST = "closest"


class RandomCircuitGenerator(AbstractCircuitGenerator):
    """Abstract base class that specifies an interface for generating new circuits
    based on a starting circuit. This includes:

        * selecting which gates to swap,
        * generating circuits based on specified gate (or other) constraints,
        * validating the generated circuits.
    """

    def __init__(
        self,
        fraction_non_clifford: float,
        method_select: str = _UNIFORM,
        method_replace: str = _CLOSEST,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
    ) -> None:
        r"""Initializer for random circuit generator.

        Args:
            fraction_non_clifford: The (approximate) fraction of non-Clifford
                gates in each returned circuit.
            method_select: Method by which non-Clifford gates are selected to
                be replaced by Clifford gates. Options are 'uniform' or
                'gaussian'.
            method_replace: Method by which selected non-Clifford gates are
                replaced by Clifford gates. Options are 'uniform', 'gaussian'
                or 'closest'.

        """
        super(RandomCircuitGenerator, self).__init__()
        self._fraction_non_clifford: float = fraction_non_clifford
        self._method_select: str = method_select
        self._method_replace: str = method_replace
        self._random_state: Optional[
            Union[int, np.random.RandomState]
        ] = random_state
        self._sigma_select: float = 0.5
        self._sigma_replace: float = 0.5
        
        if self._random_state is None or isinstance(self._random_state, int):
            self._random_state = np.random.RandomState(self._random_state)

    def _swap_operations(
        self,
        ops: Sequence[cirq.ops.Operation],
    ) -> Sequence[cirq.ops.Operation]:
        """Calls the executor function on noise-scaled quantum circuit and
        stores the results.
        """
        """Function that takes the non-Clifford angles and replacement and
        selection specifications, returning the projected angles according to a
        specific method.

        Args:
            ops: array of non-Clifford angles.
        Returns:
            rz_non_clifford_replaced: the selected non-Clifford gates replaced
                by a Clifford according to some method.

        Raises:
            Exception: If argument 'method_replace' is not either 'closest',
            'uniform' or 'gaussian'.
        """
        # TODO: Update these functions to act on operations instead of angles.
        non_clifford_angles = np.array(
            [op.gate.exponent * np.pi for op in ops]  # type: ignore
        )
        if self._method_replace == _CLOSEST:
            clifford_angles = closest_clifford(non_clifford_angles)

        elif self._method_replace == _UNIFORM:
            clifford_angles = random_clifford(len(non_clifford_angles), self._random_state)

        elif self._method_replace == _GAUSSIAN:
            clifford_angles = probabilistic_angle_to_clifford(
                non_clifford_angles, self._sigma_replace, self._random_state
            )

        else:
            raise ValueError(
                f"Arg `method_replace` must be 'closest', \
                    'uniform', or 'gaussian'"
                f" but was {self._method_replace}."
            )

        # TODO: Write function to replace the angles in a list of operations?
        return [
            cirq.ops.rz(a).on(*q)
            for (a, q) in zip(
                clifford_angles,
                [op.qubits for op in ops],
            )
        ]

    def configure_gaussian(
        self, sigma_select: float, sigma_replace: float
    ) -> None:
        """A function required to run in order to use gaussian generation."""
        if self._method_select != _GAUSSIAN:
            raise ValueError(
                f"Sigma configuration must be used with \
                    'method_select'=='gaussian'"
                f" but was {self._method_select}."
            )

        self._sigma_select = sigma_select
        self._sigma_replace = sigma_replace

    @class_atomic_one_to_many_converter
    def generate_circuits(
        self,
        circuit: Circuit,
        num_circuits_to_generate: int,
    ) -> List[Circuit]:
        r"""Returns a list of (near) Clifford circuits obtained by replacing (some)
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
        # Find the non-Clifford operations in the circuit.
        operations = np.array(list(circuit.all_operations()))
        non_clifford_indices_and_ops = np.array(
            [
                [i, op]
                for i, op in enumerate(operations)
                if not cirq.has_stabilizer_effect(op)
            ]
        )

        if len(non_clifford_indices_and_ops) == 0:
            return [circuit] * num_circuits_to_generate

        non_clifford_indices = np.int32(non_clifford_indices_and_ops[:, 0])
        non_clifford_ops = non_clifford_indices_and_ops[:, 1]

        # Replace (some of) the non-Clifford operations.
        near_clifford_circuits = []
        for _ in range(num_circuits_to_generate):
            new_ops = self._map_to_near_clifford(
                non_clifford_ops,
            )
            operations[non_clifford_indices] = new_ops
            near_clifford_circuits.append(Circuit(operations))

        return near_clifford_circuits

    def _map_to_near_clifford(
        self,
        non_clifford_ops: Sequence[cirq.ops.Operation],
    ) -> Sequence[cirq.ops.Operation]:
        """Returns the list of non-Clifford operations with some of these replaced
        by Clifford operations.

        Args:
            non_clifford_ops: A sequence of non-Clifford operations.
        """
        # Select (indices of) operations to replace.
        indices_of_selected_ops = self._select(non_clifford_ops)

        # Replace selected operations.
        clifford_ops: Sequence[cirq.ops.Operation] = self._swap_operations(
            [non_clifford_ops[i] for i in indices_of_selected_ops]
        )

        # Return sequence of (near) Clifford operations.
        return [
            cast(List[cirq.ops.Operation], clifford_ops).pop(0)
            if i in indices_of_selected_ops
            else op
            for (i, op) in enumerate(non_clifford_ops)
        ]

    def _select(
        self, non_clifford_ops: Sequence[cirq.ops.Operation]
    ) -> List[int]:
        """Returns indices of non-Clifford operations selected (to be replaced)
        according to some method.

        Args:
            non_clifford_ops: Sequence of non-Clifford operations.
        """
        num_non_cliff = len(non_clifford_ops)
        num_to_replace = int(
            round(self._fraction_non_clifford * num_non_cliff)
        )

        # Get the distribution for how to select operations.
        if self._method_select == _UNIFORM:
            distribution = (
                1.0 / num_non_cliff * np.ones(shape=(num_non_cliff,))
            )
        elif self._method_select == _GAUSSIAN:
            non_clifford_angles = np.array(
                [
                    op.gate.exponent * np.pi  # type: ignore
                    for op in non_clifford_ops
                ]
            )
            probabilities = angle_to_proximity(
                non_clifford_angles,
                self._sigma_select,
            )
            distribution = [k / sum(probabilities) for k in probabilities]
        else:
            raise ValueError(
                f"Arg `method_select` must be 'uniform' or 'gaussian' but was "
                f"{self._method_select}."
            )

        # Select (indices of) non-Clifford operations to replace.
        selected_indices = cast(
            np.random.RandomState,
            self._random_state,
        ).choice(
            range(num_non_cliff),
            num_non_cliff - num_to_replace,
            replace=False,
            p=distribution,
        )
        return [int(i) for i in sorted(selected_indices)]
