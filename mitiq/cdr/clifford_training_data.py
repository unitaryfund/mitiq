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

"""Functions for mapping circuits to near-Clifford circuits."""
from copy import deepcopy
from typing import Iterable, List, Optional, Sequence, Union

import numpy as np

import cirq
from cirq.circuits import Circuit


# Z rotation gates with these angles are Clifford gates.
_CLIFFORD_ANGLES = (0.0, np.pi / 2, np.pi, (3 / 2) * np.pi)
_CLIFFORD_EXPONENTS = np.array([0.0, 0.5, 1.0, 1.5])


def generate_training_circuits(
    circuit: Circuit,
    num_training_circuits: int,
    fraction_non_clifford: float,
    method_select: str = "uniform",
    method_replace: str = "closest",
    random_state: Optional[Union[int, np.random.RandomState]] = None,
    **kwargs: dict,
) -> List[Circuit]:
    """Returns a list of (near) Clifford circuits obtained by replacing (some)
    non-Clifford gates in the input circuit by Clifford gates.
    
    The way in which non-Clifford gates are selected to be replaced / replaced
    is determined by ``method_select`` and ``method_replace``.
    
    In the Clifford Data Regression (CDR) method [TODO: Cite paper], data
    generated from these circuits is used as a training set to learn the
    effect of noise.

    Args:
        circuit: A circuit of interest assumed to be compiled into the gate
            set {Rz, Rx, CNOT}.
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
            sigma_select (float): Width of the Gaussian distribution used for
                ``method_select='gaussian'``.
            sigma_replace (float): Width of the Gaussian distribution used for
                ``method_replace='gaussian'``.
    """
    if isinstance(random_state, int):
        random_state = np.random.RandomState(random_state)
    else:
        random_state = np.random.RandomState(np.random.randint(1e8))

    # Seeds used for each training circuit construction.
    random_states = random_state.randint(
        10_000 * num_training_circuits, size=num_training_circuits
    )

    # Find the non-Clifford operations in the circuit.
    operations = np.array(list(circuit.all_operations()))
    non_clifford_indices_and_ops = np.array(
        [[i, op] for i, op in enumerate(operations) if not _is_clifford(op)]
    )
    if len(non_clifford_indices_and_ops) == 0:
        raise ValueError("Circuit is already Clifford.")

    non_clifford_indices = np.int32(non_clifford_indices_and_ops[:, 0])
    non_clifford_ops = non_clifford_indices_and_ops[:, 1]

    # Replace (some of) the non-Clifford operations.
    near_clifford_circuits = []
    for rng in random_states:
        new_ops = _map_to_near_clifford(
            non_clifford_ops,
            fraction_non_clifford,
            method_select,
            method_replace,
            rng,
            **kwargs
        )
        operations[non_clifford_indices] = new_ops
        near_clifford_circuits.append(Circuit(operations))

    return near_clifford_circuits


def _map_to_near_clifford(
    non_clifford_ops: Sequence[cirq.ops.Operation],
    fraction_non_clifford: float,
    method_select: str = "uniform",
    method_replace: str = "closest",
    seed: Optional[int] = None,
    **kwargs: dict,
) -> Sequence[cirq.ops.Operation]:
    """Maps the input operations to a (near) Clifford circuit and returns this
    circuit.

    Args:
        non_clifford_ops: Non-Clifford operations to map to Clifford operations.
        fraction_non_clifford: The (approximate) fraction of non-Clifford
            operations in each returned circuit.
        method_select: The way in which the non-Clifford gates are selected to
            be replaced by Clifford gates. Options are 'uniform' or 'gaussian'.
        method_replace: The way in which selected non-Clifford gates are
            replaced by Clifford gates. Options are 'uniform', 'gaussian' or
            'closest'.
        seed: Seed for sampling.
        kwargs: Additional options for selection / replacement methods.
            sigma_select (float): Width of the Gaussian distribution used for
                ``method_select='gaussian'``.
            sigma_replace (float): Width of the Gaussian distribution used for
                ``method_replace='gaussian'``.

        Returns:
            Circuit: Near-Clifford projected circuit.
    """
    print("In map to near clifford...")
    sigma_select = kwargs.get("sigma_select", 0.5)
    sigma_replace = kwargs.get("sigma_replace", 0.5)

    state = np.random.RandomState(seed)
    seed_select, seed_replace = state.randint(1e8, size=2,)
    random_state_select = np.random.RandomState(seed_select)
    random_state_replace = np.random.RandomState(seed_replace)

    # Select (indices of) operations to replace.
    indices_of_selected_ops = _select(
        non_clifford_ops,
        fraction_non_clifford,
        method_select,
        sigma_select,
        random_state_select,
    )

    print("Just selected ops to replace, the indices are:")
    print(indices_of_selected_ops)

    print("The ops are:")
    print(*[non_clifford_ops[i] for i in indices_of_selected_ops], sep="\n")

    # Replace selected operations.
    clifford_ops = _replace(
        [non_clifford_ops[i] for i in indices_of_selected_ops],
        method_replace,
        sigma_replace,
        random_state_replace,
    )

    print("\nThe new ops are:")
    print(*clifford_ops, sep="\n")

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
) -> np.ndarray:
    """Returns indices of selected operations.

    Operations are selected by a strategy determined by the ``method``
    argument.

    Args:
        non_clifford_ops: Sequence of non-Clifford operations.
        fraction_non_clifford: fraction of non-Clifford gates to change.
        method: {'uniform', 'gaussian'} method to use to select Clifford gates
                to replace.
        sigma: width of probability distribution used in selection
                      of non-Clifford gates to replace, only has effect if
                      method_select = 'gaussian'.
        random_state: Random state for sampling.
    Returns:
        list of indicies that identify rotation angles to change.

    Raises:
        Exception: If argument 'method_select' is neither 'uniform' nor
                   'gaussian'.
    """
    if random_state is None:
        random_state = np.random

    num_non_cliff = len(non_clifford_ops)
    num_to_replace = int(fraction_non_clifford * num_non_cliff)

    # Get the distribution for how to select operations.
    if method == "uniform":
        distribution = 1.0 / num_non_cliff * np.ones(shape=(num_non_cliff,))
    elif method == "gaussian":
        non_clifford_angles = np.array(
            [op.gate.exponent * np.pi for op in non_clifford_ops]
        )
        probabilities = _angle_to_probabilities(non_clifford_angles, sigma)
        norm = sum(probabilities)
        distribution = [k / norm for k in probabilities]
    else:
        raise ValueError(
            f"Arg `method_select` must be 'uniform' or 'gaussian' but was "
            f"{method}."
        )
    print("Prob distribution to select gates:", distribution)

    # Select (indices of) non-Clifford operations to replace.
    indices = range(num_non_cliff)
    selected_indices = random_state.choice(
        indices, num_non_cliff - num_to_replace, replace=False, p=distribution,
    )
    # TODO: Select in a way that sorting isn't required.
    selected_indices.sort()
    return selected_indices


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
        seed: Seed for sampling.
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
        clifford_angles = _random_clifford(len(non_clifford_ops), random_state)

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
            clifford_angles,
            [op.qubits for op in non_clifford_ops],
        )
    ]


def is_clifford(op_like: cirq.ops.OP_TREE) -> bool:
    """Returns True if the input argument is Clifford, else False.

    Args:
        op_like: A single operation, list of operations, or circuit.
    """
    try:
        circuit = cirq.Circuit(op_like)
    except TypeError:
        raise ValueError("Could not convert `op_like` to a circuit.")

    return all(_is_clifford(op) for op in circuit.all_operations())


def _is_clifford(op: cirq.ops.Operation) -> bool:
    if isinstance(op.gate, cirq.ops.XPowGate):
        return True
    if isinstance(op.gate, cirq.ops.CNotPowGate) and op.gate.exponent == 1.0:
        return True
    if (
        isinstance(op.gate, cirq.ops.ZPowGate)
        and op.gate.exponent % 2 in _CLIFFORD_EXPONENTS
    ):
        return True

    # Ignore measurements.
    if isinstance(op.gate, cirq.ops.MeasurementGate):
        return True
    # TODO: Could add additional logic here.
    return False


def _project_to_closest_clifford(
    ops: Iterable[cirq.ops.Operation],
) -> List[cirq.ops.Operation]:
    clifford_ops = []
    for op in ops:
        new_exponent = _CLIFFORD_EXPONENTS[
            np.argmin(np.abs(op.gate.exponent - _CLIFFORD_EXPONENTS))
        ]
        new_op = deepcopy(op)
        new_op.gate._exponent = new_exponent
        clifford_ops.append(new_op)
    return clifford_ops


def _closest_clifford(ang: float) -> float:
    """Function to take angle and return the nearest Clifford angle note the
       usage of this function is vectorized so it takes and returns arrays.

    Args:
        ang: angle in Rz gate.

    Returns:
        Clifford angle: closest clifford angle.
    """
    ang = ang % (2 * np.pi)
    ang_scaled = ang / (np.pi / 2)
    # if just one min value, return the corresponding nearest cliff.
    if (
        abs((ang_scaled / 0.5) - 1) > 10 ** (-6)
        and abs((ang_scaled / 0.5) - 3) > 10 ** (-6)
        and (abs((ang_scaled / 0.5) - 5) > 10 ** (-6))
    ):
        index = int(np.round(ang_scaled)) % 4
        return _CLIFFORD_ANGLES[index]
    # if two min values (ie two cliff gates equidistant) randomly choose the
    # cliff gate to return.
    else:
        index_list = [ang_scaled - 0.5, ang_scaled + 0.5]
        index = int(np.random.choice(index_list))
        return _CLIFFORD_ANGLES[index]


# vectorize so function can take array of angles.
_closest_clifford = np.vectorize(_closest_clifford)


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


def count_non_cliffords(circuit: Circuit,) -> float:
    """Function to check how many non-Clifford gates are in a give circuit.

    Args:
        circuit: cirq.Circuit object already decomposed into the basis
                 {Rz, Rx(pi/2), CNOT, X}

    Returns:
        number of non-Clifford gates in the given circuit.
    """
    operations = np.array(list(circuit.all_operations()))
    gates = np.array([op.gate for op in operations])
    mask = np.array(
        [isinstance(i, cirq.ops.common_gates.ZPowGate) for i in gates]
    )
    r_z_gates = operations[mask]
    angles = np.array([op.gate.exponent * np.pi for op in r_z_gates])
    mask_non_clifford = ~_is_clifford_angle(angles)
    rz_non_clifford = angles[mask_non_clifford]
    return len(rz_non_clifford)


def _is_clifford_angle(ang: float, tol: float = 10 ** -5,) -> bool:
    """Function to check if a given angle is Clifford.
    Args:
        ang: rotation angle in the Rz gate.
    Returns:
        bool: True / False for Clifford or not.
    """
    ang = ang % (2 * np.pi)
    closest_clifford_angle = _closest_clifford(ang)
    if abs(closest_clifford_angle - ang) < tol:
        return True
    return False


# Vectorize function so it can take arrays of angles as its input.
_is_clifford_angle = np.vectorize(_is_clifford_angle)


def _angle_to_probabilities(angle: float, sigma: float) -> float:
    """Function to return probability distribution based on distance from
       angles to Clifford gates.

    Args:
        angle: angle to form probability distribution.
    Returns:
        discrete value of probability distribution calucalted from
        Prob_project = exp(-(dist/sigma)^2) where dist = sum(dists) is the
        sum of distances from each Clifford gate.
    """
    angle = angle % (2 * np.pi)
    S = np.array([[1, 0.0], [0.0, 1j]])
    Rz = np.array([[1, 0.0], [0.0, np.exp(angle * 1j)]])
    dists = []
    for i in range(4):
        if i == 0:
            i = 4
        diff = np.linalg.norm(Rz - S ** (i))
        dists.append(np.exp(-((diff / sigma) ** 2)))
    return sum(dists)


# vectorize so function can take array of angles.
_angle_to_probabilities = np.vectorize(_angle_to_probabilities)


def _probabilistic_angle_to_clifford(
    ang: float, sigma: float, random_state: np.random.RandomState
) -> float:
    """Function to take angle and return the Clifford angle according to the
       probability distirbution:

                        prob = exp(-(dist/sigma)^2)

    where dist is the frobenius norm from the 4 clifford angles and the gate
    of interest. Note the usage of this function is vectorized so it takes
    and returns arrays.

    Args:
        ang: angle in Rz gate.
        sigma: width of probability distribution.

    Returns:
        Clifford angle: clifford angle to replace gate angle, calculated
        probabilistically.
    """
    ang = ang % (2 * np.pi)
    S = np.array([[1, 0.0], [0.0, 1j]])
    Rz = np.array([[1, 0.0], [0.0, np.exp(ang * 1j)]])
    dists = []
    for i in range(4):
        if i == 0:
            i = 4
        diff = np.linalg.norm(Rz - S ** (i))
        dists.append(np.exp(-((diff / sigma) ** 2)))
    prob_gate = [i / sum(dists) for i in dists]
    cliff_ang = random_state.choice(
        _CLIFFORD_ANGLES, 1, replace=False, p=prob_gate
    )
    return cliff_ang


# vectorize so function can take array of angles.
_probabilistic_angle_to_clifford = np.vectorize(
    _probabilistic_angle_to_clifford
)
