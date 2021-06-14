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

"""Functions for mapping circuits to (near) Clifford circuits."""
from typing import List, Optional, Sequence, Union

import numpy as np

import cirq
from cirq.circuits import Circuit

from mitiq import QPROGRAM
from mitiq.interface import (
    accept_any_qprogram_as_input,
    atomic_one_to_many_converter,
)

# Z gates with these angles/exponents are Clifford gates.
_CLIFFORD_EXPONENTS = np.array([0.0, 0.5, 1.0, 1.5])
_CLIFFORD_ANGLES = [exponent * np.pi for exponent in _CLIFFORD_EXPONENTS]


@atomic_one_to_many_converter
def generate_training_circuits(
    circuit: QPROGRAM,
    num_training_circuits: int,
    fraction_non_clifford: float,
    method_select: str = "uniform",
    method_replace: str = "closest",
    random_state: Optional[Union[int, np.random.RandomState]] = None,
    **kwargs,
) -> List[QPROGRAM]:
    r"""Returns a list of (near) Clifford circuits obtained by replacing (some)
    non-Clifford gates in the input circuit by Clifford gates.

    The way in which non-Clifford gates are selected to be replaced is
    determined by ``method_select`` and ``method_replace``.

    In the Clifford Data Regression (CDR) method [Czarnik2020]_, data
    generated from these circuits is used as a training set to learn the
    effect of noise.

    Args:
        circuit: A circuit of interest assumed to be compiled into the gate
            set {Rz, sqrt(X), CNOT}, or such that all the non-Clifford gates
            are contained in the Rz rotations.
        num_training_circuits: Number of circuits in the returned training set.
        fraction_non_clifford: The (approximate) fraction of non-Clifford
            gates in each returned circuit.
        method_select: Method by which non-Clifford gates are selected to be
            replaced by Clifford gates. Options are 'uniform' or 'gaussian'.
        method_replace: Method by which selected non-Clifford gates are
            replaced by Clifford gates. Options are 'uniform', 'gaussian' or
            'closest'.
        random_state: Seed for sampling.
        kwargs: Available keyword arguments are:
            - sigma_select (float): Width of the Gaussian distribution used for
            ``method_select='gaussian'``.
            - sigma_replace (float): Width of the Gaussian distribution used
            for ``method_replace='gaussian'``.

    .. [Czarnik2020] : Piotr Czarnik, Andrew Arramsmith, Patrick Coles,
        Lukasz Cincio, "Error mitigation with Clifford quantum circuit
        data," (https://arxiv.org/abs/2005.10189).
    """
    if random_state is None or isinstance(random_state, int):
        random_state = np.random.RandomState(random_state)

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
        raise ValueError("Circuit is already Clifford.")

    non_clifford_indices = np.int32(non_clifford_indices_and_ops[:, 0])
    non_clifford_ops = non_clifford_indices_and_ops[:, 1]

    # Replace (some of) the non-Clifford operations.
    near_clifford_circuits = []
    for _ in range(num_training_circuits):
        new_ops = _map_to_near_clifford(
            non_clifford_ops,
            fraction_non_clifford,
            method_select,
            method_replace,
            random_state,
            **kwargs,
        )
        operations[non_clifford_indices] = new_ops
        near_clifford_circuits.append(Circuit(operations))

    return near_clifford_circuits


@accept_any_qprogram_as_input
def is_clifford(circuit: QPROGRAM) -> bool:
    """Returns True if the input argument is Clifford, else False.

    Args:
        circuit: A single operation, list of operations, or circuit.
    """
    return all(
        cirq.has_stabilizer_effect(op) for op in circuit.all_operations()
    )


@accept_any_qprogram_as_input
def count_non_cliffords(circuit: QPROGRAM) -> int:
    """Returns the number of non-Clifford operations in the circuit. Assumes
    the circuit consists of only Rz, Rx, and CNOT operations.

    Args:
        circuit: Circuit to count the number of non-Clifford operations in.
    """
    return sum(
        not cirq.has_stabilizer_effect(op) for op in circuit.all_operations()
    )


def _map_to_near_clifford(
    non_clifford_ops: Sequence[cirq.ops.Operation],
    fraction_non_clifford: float,
    method_select: str = "uniform",
    method_replace: str = "closest",
    random_state: Optional[np.random.RandomState] = None,
    **kwargs: dict,
) -> Sequence[cirq.ops.Operation]:
    """Returns the list of non-Clifford operations with some of these replaced
    by Clifford operations.

    Args:
        non_clifford_ops: A sequence of non-Clifford operations.
        fraction_non_clifford: The (approximate) fraction of non-Clifford
            operations in the returned list.
        method_select: The way in which the non-Clifford gates are selected to
            be replaced by Clifford gates. Options are 'uniform' or 'gaussian'.
        method_replace: The way in which selected non-Clifford gates are
            replaced by Clifford gates. Options are 'uniform', 'gaussian' or
            'closest'.
        random_state: Seed for sampling.
        kwargs: Additional options for selection / replacement methods.
            sigma_select (float): Width of the Gaussian distribution used for
                ``method_select='gaussian'``.
            sigma_replace (float): Width of the Gaussian distribution used for
                ``method_replace='gaussian'``.
    """
    sigma_select = kwargs.get("sigma_select", 0.5)
    sigma_replace = kwargs.get("sigma_replace", 0.5)

    # Select (indices of) operations to replace.
    indices_of_selected_ops = _select(
        non_clifford_ops,
        fraction_non_clifford,
        method_select,
        sigma_select,
        random_state,
    )

    # Replace selected operations.
    clifford_ops = _replace(
        [non_clifford_ops[i] for i in indices_of_selected_ops],
        method_replace,
        sigma_replace,
        random_state,
    )

    # Return sequence of (near) Clifford operations.
    return [
        clifford_ops.pop(0) if i in indices_of_selected_ops else op
        for (i, op) in enumerate(non_clifford_ops)
    ]


def _select(
    non_clifford_ops: Sequence[cirq.ops.Operation],
    fraction_non_clifford: float,
    method: str = "uniform",
    sigma: Optional[float] = 1.0,
    random_state: Optional[np.random.RandomState] = None,
) -> List[int]:
    """Returns indices of non-Clifford operations selected (to be replaced)
    according to some method.

    Args:
        non_clifford_ops: Sequence of non-Clifford operations.
        fraction_non_clifford: fraction of non-Clifford gates to change.
        method: {'uniform', 'gaussian'} method to use to select Clifford gates
                to replace.
        sigma: width of probability distribution used in selection
                      of non-Clifford gates to replace, only has effect if
                      method_select = 'gaussian'.
        random_state: Random state for sampling.
    """
    if random_state is None:
        random_state = np.random

    num_non_cliff = len(non_clifford_ops)
    num_to_replace = int(round(fraction_non_clifford * num_non_cliff))

    # Get the distribution for how to select operations.
    if method == "uniform":
        distribution = 1.0 / num_non_cliff * np.ones(shape=(num_non_cliff,))
    elif method == "gaussian":
        non_clifford_angles = np.array(
            [op.gate.exponent * np.pi for op in non_clifford_ops]
        )
        probabilities = _angle_to_proximity(non_clifford_angles, sigma)
        distribution = [k / sum(probabilities) for k in probabilities]
    else:
        raise ValueError(
            f"Arg `method_select` must be 'uniform' or 'gaussian' but was "
            f"{method}."
        )

    # Select (indices of) non-Clifford operations to replace.
    selected_indices = random_state.choice(
        range(num_non_cliff),
        num_non_cliff - num_to_replace,
        replace=False,
        p=distribution,
    )
    return [int(i) for i in sorted(selected_indices)]


def _replace(
    non_clifford_ops: Sequence[cirq.ops.Operation],
    method: str = "uniform",
    sigma: float = 1.0,
    random_state: Optional[np.random.RandomState] = None,
) -> Sequence[cirq.ops.Operation]:
    """Function that takes the non-Clifford angles and replacement and
    selection specifications, returning the projected angles according to a
    specific method.

    Args:
        non_clifford_ops: array of non-Clifford angles.
        method: {'uniform', 'gaussian', 'closest'} method to use
                        to replace selected non-Clifford gates.
        sigma: width of probability distribution used in replacement
                       of selected non-Clifford gates, only has effect if
                       method_replace = 'gaussian'.
        random_state: Seed for sampling.
    Returns:
        rz_non_clifford_replaced: the selected non-Clifford gates replaced by a
                               Clifford according to some method.

    Raises:
        Exception: If argument 'method_replace' is not either 'closest',
        'uniform' or 'gaussian'.
    """
    if random_state is None:
        random_state = np.random

    # TODO: Update these functions to act on operations instead of angles.
    non_clifford_angles = np.array(
        [op.gate.exponent * np.pi for op in non_clifford_ops]
    )
    if method == "closest":
        clifford_angles = _closest_clifford(non_clifford_angles)

    elif method == "uniform":
        clifford_angles = _random_clifford(
            len(non_clifford_angles), random_state
        )

    elif method == "gaussian":
        clifford_angles = _probabilistic_angle_to_clifford(
            non_clifford_angles, sigma, random_state
        )

    else:
        raise ValueError(
            f"Arg `method_replace` must be 'closest', 'uniform', or 'gaussian'"
            f" but was {method}."
        )

    # TODO: Write function to replace the angles in a list of operations?
    return [
        cirq.ops.rz(a).on(*q)
        for (a, q) in zip(
            clifford_angles, [op.qubits for op in non_clifford_ops],
        )
    ]


def _random_clifford(
    num_angles: int, random_state: np.random.RandomState
) -> np.ndarray:
    """Returns an array of Clifford angles chosen uniformly at random.

    Args:
        num_angles: Number of Clifford angles to return in array.
        random_state: Random state for sampling.
    """
    return np.array(
        [random_state.choice(_CLIFFORD_ANGLES) for _ in range(num_angles)]
    )


@np.vectorize
def _closest_clifford(angles: np.ndarray) -> float:
    """Returns the nearest Clifford angles to the input angles.

    Args:
        non_Clifford_ops: Non-Clifford opperations.
    """
    ang_scaled = angles / (np.pi / 2)
    # if just one min value, return the corresponding nearest cliff.
    if (
        abs((ang_scaled / 0.5) - 1) > 10 ** (-6)
        and abs((ang_scaled / 0.5) - 3) > 10 ** (-6)
        and (abs((ang_scaled / 0.5) - 5) > 10 ** (-6))
    ):
        index = int(np.round(ang_scaled)) % 4
        return _CLIFFORD_ANGLES[index]
    # If equidistant between two Clifford angles, randomly choose one.
    else:
        index_list = [ang_scaled - 0.5, ang_scaled + 0.5]
        index = int(np.random.choice(index_list))
        return _CLIFFORD_ANGLES[index]


@np.vectorize
def _is_clifford_angle(angles: np.ndarray, tol: float = 10 ** -5,) -> bool:
    """Function to check if a given angle is Clifford.

    Args:
        angles: rotation angle in the Rz gate.
    """
    angles = angles % (2 * np.pi)
    closest_clifford_angle = _closest_clifford(angles)
    if abs(closest_clifford_angle - angles) < tol:
        return True
    return False


def _angle_to_proximities(angle: np.ndarray, sigma: float) -> np.ndarray:
    """Returns probability distribution based on distance from angles to
    Clifford gates.

    Args:
        angle: angle to form probability distribution.

    Returns:
        discrete value of probability distribution calculated from
        exp(-(diff/sigma)^2) where diff is the distance from each angle and the
        Clifford gates.
    """
    s_matrix = cirq.unitary(cirq.S)
    rz_matrix = cirq.unitary(cirq.rz(angle % (2 * np.pi)))
    # TODO: Update loop / if.
    dists = []
    for exponent in range(4):
        if exponent == 0:
            exponent = 4
        diff = np.linalg.norm(rz_matrix - s_matrix ** exponent)
        dists.append(np.exp(-((diff / sigma) ** 2)))
    return dists


@np.vectorize
def _angle_to_proximity(angle: np.ndarray, sigma: float) -> float:
    """Returns probability distribution based on distance from angles to
    Clifford gates.

    Args:
        angle: angle to form probability distribution.

    Returns:
        discrete value of probability distribution calculated from
        exp(-(dist/sigma)^2) where dist = sum(dists) is the
        sum of distances from each Clifford gate.
    """
    dists = _angle_to_proximities(angle, sigma)
    return np.max(dists)


@np.vectorize
def _probabilistic_angle_to_clifford(
    angles: np.ndarray, sigma: float, random_state: np.random.RandomState,
) -> float:
    """Returns a Clifford angle sampled from the distribution

                        prob = exp(-(dist/sigma)^2)

    where dist is the Frobenius norm from the 4 clifford angles and the gate
    of interest.

    Args:
        angles: Non-Clifford angles.
        sigma: Width of probability distribution.
    """

    dists = _angle_to_proximities(angles, sigma)

    cliff_ang = random_state.choice(
        _CLIFFORD_ANGLES, 1, replace=False, p=np.array(dists) / np.sum(dists)
    )
    return cliff_ang
