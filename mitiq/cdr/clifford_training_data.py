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
import cirq
from cirq.circuits import Circuit
import numpy as np
from typing import List, Tuple, Optional, Union


# Global variable of Clifford angles in Rz gates:
CLIFFORD_ANGLES = (0.0, np.pi / 2, np.pi, (3 / 2) * (np.pi))


def generate_training_circuits(
    circuit: Circuit,
    num_training_circuits: int,
    fraction_non_clifford: float,
    method_select: str = "uniform",
    method_replace: str = "closest",
    random_state: Optional[Union[int, np.random.RandomState]] = None,
    **additional_options: dict,
) -> Tuple[List[Circuit], List[List[float]], List[List[float]]]:

    """Function to return a list of near-Clifford circuits from one circuit of
    interest. Data generated from these circuits is used as a training set to
    learn the effect of noise.

    Args:
        circuit: A circuit of interest, assumes already compiled into gate set
                               (Rz, Rx, Z, X, CNOT).
        num_training_circuits: Number of circuits in the returned training set.
        fraction_non_clifford: The (approximate) fraction of non-Clifford
                               gates in each returned circuit.
        method_select: option to define the way in which the non-Clifford
                              gates to replace with Cliffords are chosen.
                              Options are 'uniform' or 'gaussian'.
        method_replace: str = option to define the way the chosen non-Clifford
                              gates are replace with a Clifford gate can take
                              strings 'uniform', 'gaussian' or 'closest'.
        random_state: Seed for sampling.
        additional_options: sequence of key = value pairs with keys defined as:
            'sigma_select': float -  postitive variable definined width of
            probability distribution used in choosing
            which non-Cliffords to replace, only has
            an impact if method_select = 'gaussian'.
            'sigma_replace': float - positive variable which defines the width
            of probability distribution used in
            choosing which Clifford to replace the
            non-Clifford with, only has an impact if
            method_replace = 'gaussian'.

    Returns:
        List[cirq.Circuit]: list of near-Clifford circuits constructed from
                             the circuits of interest.
        List[List[float]]: list of list of angles that were replaced in each
                           training circuit.
        List[List[float]]: list of list of angles that were inserted in each
                           training circuit.
    """
    circuits_list = []
    # setting the seed:
    if isinstance(random_state, int):
        random_state = np.random.RandomState(random_state)
    else:
        random_state = np.random.randint(10 ** (8))
        random_state = np.random.RandomState(random_state)
    # generating a list of seeds used for each training circuit construction:
    random_states = random_state.randint(
        10000 * num_training_circuits, size=num_training_circuits
    )
    # get the operations from the circuit and find the non-cliff angles:
    operations = np.array(list(circuit.all_operations()))
    gates = np.array([op.gate for op in operations])
    qubits = np.array([op.qubits[0] for op in operations])
    positions = np.array(range(len(gates)))
    zgatesmask = np.array(
        [isinstance(gate, cirq.ops.common_gates.ZPowGate) for gate in gates]
    )
    r_z_gates = operations[zgatesmask]
    r_z_positions = positions[zgatesmask]
    r_z_qubits = qubits[zgatesmask]
    angles = _get_arguments(r_z_gates)
    mask_non_clifford = ~_is_clifford_angle(angles)
    rz_non_clifford = angles[mask_non_clifford]
    position_non_clifford = r_z_positions[mask_non_clifford]
    qubits_non_clifford = r_z_qubits[mask_non_clifford]
    for n in range(num_training_circuits):
        random_state = random_states[n]
        random_state = np.random.RandomState(random_state)
        #  array:
        if additional_options:
            (projected_circuit) = _map_to_near_clifford(
                operations.copy(),
                rz_non_clifford.copy(),
                position_non_clifford.copy(),
                qubits_non_clifford.copy(),
                fraction_non_clifford,
                random_state,
                method_select,
                method_replace,
                additional_options=additional_options.get(
                    "additional_options"
                ),
            )
        else:
            (projected_circuit) = _map_to_near_clifford(
                operations.copy(),
                rz_non_clifford.copy(),
                position_non_clifford.copy(),
                qubits_non_clifford.copy(),
                fraction_non_clifford,
                random_state,
                method_select,
                method_replace,
            )
        circuits_list.append(projected_circuit)
    return circuits_list


def _map_to_near_clifford(
    operations: np.ndarray,
    rz_non_clifford: np.ndarray,
    position_non_clifford: np.ndarray,
    qubits_non_clifford: np.ndarray,
    fraction_non_clifford: float,
    random_state: np.random.RandomState,
    method_select: str = "uniform",
    method_replace: str = "closest",
    **additional_options: dict,
) -> Tuple[Circuit, List[float], List[float]]:
    """Function to take the information in some circuit of interest and
        return a near-Clifford circuit as constructed according to
        some user defined methods.
    Args:
        operations: Cirq operations that make up the circuit of interest.
        rz_non_clifford: non-Clifford Rz gates in the circuit of interest.
        position_non_clifford: non-Clifford positions for circuit of interest.
        qubits_non_clifford: qubits that non-Clifford gates act on in circuit
                             of interest.
        fraction_non_clifford: the fraction of non-Clifford gates to replace
                               in the circuit of interest.
        method_select: {'uniform', 'gaussian'} method to use to select
                       non-Clifford gates to replace.
        method_replace: {'uniform', 'gaussian', 'closest'} method to use to
                        replace selected non-Clifford.
        random_state: Seed for sampling.
        sigma_select: width of probability distribution used in selection
                      of non-Clifford gates to replace, only has effect if
                      method_select = 'gaussian'
        sigma_replace: width of probability distribution used in replacement
                       of selected non-Clifford gates, only has effect if
                       method_replace = 'gaussian'.
        Returns:
            Circuit: Near-Clifford projected circuit.
    """
    additional_options = additional_options.get("additional_options")
    # testing the dictionary exists:
    if bool(additional_options):
        sigma_select = additional_options.setdefault("sigma_select", 0.5)
        sigma_replace = additional_options.setdefault("sigma_replace", 0.5)
    else:
        additional_options = {}
        sigma_select = additional_options.setdefault("sigma_select", 0.5)
        sigma_replace = additional_options.setdefault("sigma_replace", 0.5)
    sigma_select = additional_options.get("sigma_select")
    sigma_replace = additional_options.get("sigma_replace")
    # set the seed for sampling, for replacement and selection:
    (random_state_select, random_state_replace) = random_state.randint(
        10 ** (8), size=2
    )
    random_state_select = np.random.RandomState(random_state_select)
    random_state_replace = np.random.RandomState(random_state_replace)
    # Choose non Clifford gates to change according to selection methods:
    columns_to_change = _select(
        rz_non_clifford,
        fraction_non_clifford,
        method_select,
        sigma_select,
        random_state_select,
    )
    rz_non_clifford_selected = rz_non_clifford[columns_to_change]
    position_selected = position_non_clifford[columns_to_change]
    qubits_selected = qubits_non_clifford[columns_to_change]
    # Now the non Clifford gates have been selected, we need to decide which
    # Clifford gate to replace them with.
    rz_non_clifford_replaced = _replace(
        rz_non_clifford_selected,
        method_replace,
        sigma_select,
        sigma_replace,
        random_state_replace,
    )
    new_gates = [
        cirq.ops.rz(parameter) for parameter in rz_non_clifford_replaced
    ]
    # rebulding the operations:
    new_operations = []
    for i, gate in enumerate(new_gates):
        qubit = qubits_selected[i]
        new_operations.append(gate(qubit))
    operations[[int(i) for i in position_selected]] = new_operations
    projected_circuit = Circuit(operations)
    return projected_circuit


def _get_arguments(operation: cirq.ops.GateOperation) -> float:
    """Takes a cirq GateOperation object and returns the exponent multiplied
    by pi. This corresponds to the angle of a rotation gate.

    Args:
        operation: a cirq GateOperation.

    Returns:
        operation.gate.exponent*pi: cirq.ops.GateOperation.gate.exponent*pi"""
    return operation.gate.exponent * np.pi


_get_arguments = np.vectorize(_get_arguments)


def _select(
    rz_non_clifford: np.ndarray,
    fraction_non_clifford: float,
    method_select: str,
    sigma_select: float,
    random_state: np.random.RandomState,
) -> np.ndarray:
    """Function to select the non-Clifford gates to be replaced for a given set
    of  non-Clifford gates.

    Args:
        rz_non_clifford: array of non-Clifford angles for a circuit of
                         interest.
        fration_non_clifford: fraction of non-Clifford gates to change.
        method: {'uniform', 'gaussian'} method to use to select Clifford gates
                to replace.
        sigma_select: width of probability distribution used in selection
                      of non-Clifford gates to replace, only has effect if
                      method_select = 'gaussian'.
        random_state: Seed for sampling.
    Returns:
        list of indicies that identify rotation angles to change.

    Raises:
        Exception: If argument 'method_select' is not either 'uniform' or
                   'gaussian'.
    """
    total_non_cliff = len(rz_non_clifford)
    N = int(fraction_non_clifford * total_non_cliff)
    if method_select == "uniform":
        columns_to_change = random_state.choice(
            np.arange(0, total_non_cliff, 1).tolist(),
            total_non_cliff - N,
            replace=False,
        )
    elif method_select == "gaussian":
        non_cliff_angles = rz_non_clifford
        # form a probability distribution:
        probabilities = _angle_to_probabilities(non_cliff_angles, sigma_select)
        prob_choose_gate = [k / sum(probabilities) for k in probabilities]
        columns_to_change = random_state.choice(
            np.arange(0, total_non_cliff, 1).tolist(),
            total_non_cliff - N,
            replace=False,
            p=prob_choose_gate,
        )
    else:
        raise Exception(
            f"Arg `method_select` must be 'uniform', or \
            'gaussian' but was {method_select}"
        )
    columns_to_change.sort()
    return columns_to_change


def _replace(
    rz_non_clifford_selected: np.ndarray,
    method_replace: str,
    sigma_select: float,
    sigma_replace: float,
    random_state: np.random.RandomState,
) -> np.ndarray:
    """Function that takes the non-Clifford angles and replacement and
    selection specifications, returning the projected angles according to a
    specific method.

    Args:
        rz_non_clifford_selected: array of non-Clifford angles.
        method_replace: {'uniform', 'gaussian', 'closest'} method to use
                        to replace selected non-Clifford gates.
        sigma_select: width of probability distribution used in selection
                      of non-Clifford gates to replace, only has effect if
                      method_select = 'gaussian'
        sigma_replace: width of probability distribution used in replacement
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
    if method_replace == "closest":
        rz_non_clifford_replaced = _closest_clifford(rz_non_clifford_selected)

    elif method_replace == "uniform":
        rz_non_clifford_replaced = _random_clifford(
            rz_non_clifford_selected, random_state
        )

    elif method_replace == "gaussian":
        rz_non_clifford_replaced = _probabilistic_angle_to_clifford(
            rz_non_clifford_selected, sigma_replace, random_state
        )
    else:
        raise Exception(
            f"Arg `method_replace` must be 'closest', 'uniform', or \
                'gaussian' but was {method_replace}"
        )
    return rz_non_clifford_replaced


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
    angles = _get_arguments(r_z_gates)
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
        return CLIFFORD_ANGLES[index]
    # if two min values (ie two cliff gates equidistant) randomly choose the
    # cliff gate to return.
    else:
        index_list = [ang_scaled - 0.5, ang_scaled + 0.5]
        index = int(np.random.choice(index_list))
        return CLIFFORD_ANGLES[index]


# vectorize so function can take array of angles.
_closest_clifford = np.vectorize(_closest_clifford)


def _random_clifford(ang: float, random_state: np.random.RandomState) -> float:
    """Function to take angle and return the random Clifford angle note the
       usage of this function is vectorized so it takes and returns arrays.

    Args:
        random_state: np.random.RandomState

    Returns:
        Clifford angle: closest clifford angle.
    """
    random_index = random_state.randint(0, 3)
    clifford_angle = CLIFFORD_ANGLES[random_index]
    return clifford_angle


# vectorize so function can take array:
_random_clifford = np.vectorize(_random_clifford)


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
        CLIFFORD_ANGLES, 1, replace=False, p=prob_gate
    )
    return cliff_ang


# vectorize so function can take array of angles.
_probabilistic_angle_to_clifford = np.vectorize(
    _probabilistic_angle_to_clifford
)
