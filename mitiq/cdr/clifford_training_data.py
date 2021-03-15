import cirq
from cirq.circuits import Circuit
from numpy.random import choice, randint
import numpy as np
from typing import List, Tuple, Optional, Union


# Global variable of Clifford angles in Rz gates:
CLIFFORD_ANGLES = (0.0, np.pi/2, np.pi, (3/2)*(np.pi))


def generate_training_circuits(
    circuit: Circuit,
    num_training_circuits: int,
    fraction_non_clifford: float,
    method_select: str = 'random',
    method_replace: str = 'closest',
    random_state: Optional[Union[int, np.random.RandomState]] = None,
    **additional_options: dict
) -> Tuple[List[Circuit], List[List[float]], List[List[float]]]:
    """Function to return a list of near-Clifford circuits to create the
    training data.

    Args:
        circuit: A circuit of interest.
        num_training_circuits: Number of circuits in the returned training set,
                               assumes already compiled into gate set
                               (Rz, Rx, Z, X, CNOT)
        fraction_non_clifford: The (approximate) fraction of non-Clifford
                               gates in each returned circuit.
        method_select: option to define the way in which the non-Clifford
                              gates to replace with Cliffords are chosen can
                              take strings 'random' or 'probabilistic'.
        method_replace: str = option to define the way the chosen non-Clifford
                              gates are replace with a Clifford gate can take
                              strings 'random', 'probabilistic' or 'closest'.
        random_state: Seed for sampling.
        additional_options: dictionary with the following keys and values:
            'sigma_select': float -  postitive variable definined width of
                                     probability distribution used in choosing
                                     which non-Cliffords to replace, only has
                                     an impact if
                                     method_select = 'probabilistic'.
            'sigma_replace': float - positive variable definined width of
                                     probability distribution used in choosing
                                     which Clifford to replace the
                                     non-Clifford with, only has an impact if
                                     method_replace = 'probabilistic'.
    Returns:
        List[circ.Circuits]: list of near-Clifford circuits constructed from
                             the circuits of interest.
        List[List[float]]: list of list of angles that were replaced in each
                           training circuit.
        List[List[float]]: list of list of angles that were inserted in each
                           training circuit.
    """
    circuits_list = []
    # find all the non-Clifford gates:
    # all_cliff = np.column_stack((not_rz_circ_data, rz_cliff))
    angles_original_list = []
    angles_replaced_list = []
    # setting the seed:
    if isinstance(random_state, int):
        np.random.seed = random_state
    else:
        random_state = randint(10**(8))
        np.random.seed = random_state
    # generating a list of seeds used for each trianing circuit construction:
    random_states = np.random.randint(10000*num_training_circuits,
                                      size=num_training_circuits)
    for n in range(num_training_circuits):
        random_state = random_states[n]
        # Convert data arry into cirq circuit and append it to the storage
        #  array:
        if additional_options:
            (projected_circuit, angles_original,
                angles_replaced) = _map_to_near_clifford(
                circuit,
                fraction_non_clifford,
                random_state,
                method_select,
                method_replace,
                additional_options=additional_options.get(
                    'additional_options'))
        else:
            (projected_circuit, angles_original,
                angles_replaced) = _map_to_near_clifford(circuit,
                                                         fraction_non_clifford,
                                                         random_state,
                                                         method_select,
                                                         method_replace)

        circuits_list.append(projected_circuit)
        # this information is to make sure the probabilistic methods are
        #  working as expected:
        angles_original_list.append(angles_original)
        angles_replaced_list.append(angles_replaced)
    return circuits_list, angles_original_list, angles_replaced_list


def _map_to_near_clifford(
    circuit: Circuit,
    fraction_non_clifford: float,
    random_state: Union[int, np.random.RandomState],
    method_select: str = 'random',
    method_replace: str = 'closest',
    **additional_options: dict
) -> Tuple[Circuit, List[float], List[float]]:
    """ Function to take the information in some circuit of interest and
        return a near-Clifford circuit as constructed according to
        some user defined methods.
    Args:
        circuit: cirq.Circuit object of origional circuit of interest.
        fraction_non_clifford: the fraction of non-Clifford gates to replace
                               in the circuit of interest.
        method_select: string defining the method used to select the
                       non-Clifford gates to replace: 'random' or
                       'probabilistic'.
        method_replace: string defining method used to replace selected
                        non-Clifford gates: 'closest', 'random' or
                        'probabilistic'.
        random_state: Seed for sampling.
        sigma_select: width of probability distribution used in selection
                      of non-Clifford gates to replace, only has effect if
                      method_select = 'probabilistic'
        sigma_replace: width of probability distribution used in replacement
                       of selected non-Clifford gates, only has effect if
                       method_replace = 'probabilistic'.
        Returns:
            Circuit: Near-Clifford projected circuit.
            List[float]: list of angles replaced that were replaced in the
                         training circuit.
            List[float]: list of angles that were inserted in the training
                         circuit.
    Raises:
        Exception: If additional options does not contain one or two keys
                   'sigma_select' and/or 'sigma_replace' both equal to some
                   float.
        .
    """
    # set the seed for sampling, for replacement and selection:
    np.random.seed = random_state
    (random_state_select, random_state_replace) = randint(10**(8),
                                                          size=2)
    # get the operations from the circuit and find the non-cliff angles:
    operations = np.array(list(circuit.all_operations()))
    positions = np.linspace(1, len(operations), len(operations))
    gates = _get_gates(operations)
    mask = np.array(
        [isinstance(i, cirq.ops.common_gates.ZPowGate) for i in gates])
    r_z_gates = operations[mask]
    r_z_positions = positions[mask]
    angles = _get_arguments(r_z_gates)
    mask_non_cliff = ~_is_clifford_angle(angles)
    rz_non_cliff = angles[mask_non_cliff]
    pos_non_cliff = r_z_positions[mask_non_cliff]
    rz_non_cliff_copy = rz_non_cliff.copy()
    sigma_select = additional_options.setdefault("sigma_select", 0.5)
    sigma_replace = additional_options.setdefault("sigma_replace", 0.5)
    if ('sigma_select' not in additional_options
            and 'sigma_replace' not in additional_options):
        raise Exception('additional options must be dicitonary with \
                        keys containing one or both of '
                        '"sigma_select" and "sigma_replace" both \
                        equal to some positive float')
    # Choose non Clifford gates to change according to selection methods:
    columns_to_change = _select(rz_non_cliff_copy, fraction_non_clifford,
                                method_select, sigma_select,
                                random_state_select)
    rz_non_cliff_selected = rz_non_cliff_copy[columns_to_change]
    pos_selected = pos_non_cliff[columns_to_change]
    # Now the non Clifford gates have been selected, we need to decide which
    # Clifford gate to replace them with.
    # to store original angles replaced:
    angles_original = rz_non_cliff_selected.copy()
    rz_non_cliff_selected = _replace(rz_non_cliff_selected, method_replace,
                                     sigma_select, sigma_replace,
                                     random_state_replace)
    # to store replaced angles:
    angles_replaced = rz_non_cliff_selected.copy()
    # build projected cirucit:
    projected_circuit = circuit.copy()[0:0]
    count = 0
    for o, op in enumerate(operations):
        if (o+1) in pos_selected:
            qubit = op.qubits[0]
            parameter = rz_non_cliff_selected[count]
            operation = cirq.ops.rz(parameter)
            projected_circuit.append(operation(qubit))
            count += 1
        else:
            projected_circuit.append(op)
    return projected_circuit, angles_original, angles_replaced


def _get_gates(
    operation: cirq.ops.GateOperation
) -> float:
    """ Takes a cirq GateOperation object and returns the gate.

    Args:
        operation: a cirq GateOperation.

    Returns:
        operation.gate: cirq.ops.GateOperation.gate"""
    return(operation.gate)


_get_gates = np.vectorize(_get_gates)


def _get_arguments(
    operation: cirq.ops.GateOperation
) -> float:
    """ Takes a cirq GateOperation object and returns the exponent multiplied
    by pi. This corresponds to the angle of a rotation gate.

    Args:
        operation: a cirq GateOperation.

    Returns:
        operation.gate.exponent*pi: cirq.ops.GateOperation.gate.exponent*pi"""
    return(operation.gate.exponent*np.pi)


_get_arguments = np.vectorize(_get_arguments)


def _select(
    rz_non_cliff: np.ndarray,
    fraction_non_clifford: float,
    method_select: str,
    sigma_select: float,
    random_state: Union[int, np.random.RandomState]
) -> np.ndarray:
    """Function to select the non-Clifford gates to be replace for a given set
    of  non-Clifford gates.

    Args:
        rz_non_cliff: array of non-Clifford angles for a circuit of interest.
        fration_non_clifford: fraction of non-Clifford gates to change.
        method_select: string specifying method to use to select Clifford gates
                       to replace can be 'random' or 'probabilistic.
        sigma_select: width of probability distribution used in selection
                      of non-Clifford gates to replace, only has effect if
                      method_select = 'probabilistic'.
        random_state: Seed for sampling.
    Returns:
        list of indicies that identify rotation angles to change.

    Raises:
        Exception: If argument 'method_select' is not either 'random' or
                   'probabilistic'.
    """
    # seeding:
    np.random.seed = random_state
    total_non_cliff = len(rz_non_cliff)
    N = int(fraction_non_clifford * total_non_cliff)
    if method_select == 'random':
        columns_to_change = choice(
            np.arange(0, total_non_cliff, 1).tolist(), total_non_cliff-N,
            replace=False)
    elif method_select == 'probabilistic':
        non_cliff_angles = rz_non_cliff
        # form a probability distribution:
        probabilities = _angle_to_probabilities(non_cliff_angles, sigma_select)
        prob_choose_gate = [k / sum(probabilities) for k in probabilities]
        columns_to_change = choice(
            np.arange(0, total_non_cliff, 1).tolist(),
            total_non_cliff - N, replace=False, p=prob_choose_gate)
    else:
        raise Exception(f"Arg `method_select` must be 'random', or \
            'probabilistic' but was {method_select}")
    columns_to_change.sort()
    return columns_to_change


def _replace(
    rz_non_cliff_selected: np.ndarray,
    method_replace: str,
    sigma_select: float,
    sigma_replace: float,
    random_state: Union[int, np.random.RandomState],
) -> np.ndarray:
    """Function that takes the non-Clifford angles and replacement and selection
    specifications, returning the projected angles according to a specific
    method.

    Args:
        rz_non_cliff_selected: array of non-Clifford angles.
        method_replace: string either 'closest', 'random' or 'probabilistic'
                        that specifies the replacement method.
        sigma_select: width of probability distribution used in selection
                      of non-Clifford gates to replace, only has effect if
                      method_select = 'probabilistic'
        sigma_replace: width of probability distribution used in replacement
                       of selected non-Clifford gates, only has effect if
                       method_replace = 'probabilistic'.
        random_state: Seed for sampling.
    Returns:
        rz_non_cliff_selected: the selected non-Clifford gates replaced by a
                               Clifford according to some method.

    Raises:
        Exception: If argument 'method_replace' is not either 'closest',
        'random' or 'probabilistic'.
    """
    # seeding:
    np.random.seed = random_state
    if method_replace == 'closest':
        rz_non_cliff_selected = _closest_clifford(
            rz_non_cliff_selected)

    elif method_replace == 'random':
        rz_non_cliff_selected = _random_clifford(
            rz_non_cliff_selected)

    elif method_replace == 'probabilistic':
        rz_non_cliff_selected = _probabilistic_angle_to_clifford(
            rz_non_cliff_selected, sigma_replace)

    else:
        raise Exception(
            f"Arg `method_replace` must be 'closest', 'random', or \
                'probabilistic' but was {method_replace}")

    return rz_non_cliff_selected


def count_non_cliffords(
    circuit: Circuit,
) -> float:
    """Function to check how many non-Clifford gates are in a give circuit.

    Args:
        circuit: cirq.Circuit object already decomposed into the basis
                 {Rz, Rx(pi/2), CNOT, X}

    Returns:
        number of non-Clifford gates in the given circuit.
    """
    operations = np.array(list(circuit.all_operations()))
    gates = _get_gates(operations)
    mask = np.array(
        [isinstance(i, cirq.ops.common_gates.ZPowGate) for i in gates])
    r_z_gates = operations[mask]
    angles = _get_arguments(r_z_gates)
    mask_non_cliff = ~_is_clifford_angle(angles)
    rz_non_cliff = angles[mask_non_cliff]
    return len(rz_non_cliff)


def _is_clifford_angle(
    ang: float,
    tol: float = 10 ** -5,
) -> bool:
    """Function to check if a given angle is Clifford.
    Args:
        ang: rotation angle in the Rz gate.
    Returns:
        bool: True / False for Clifford or not.
    """
    ang = ang % (2*np.pi)
    closest_clifford_angle = _closest_clifford(ang)
    if abs(closest_clifford_angle - ang) < tol:
        return True
    else:
        return False


# Vectorize function so it can take arrays of angles as its input.
_is_clifford_angle = np.vectorize(_is_clifford_angle)


def _closest_clifford(
    ang: float
) -> float:
    """Function to take angle and return the nearest Clifford angle note the
       usage of this function is vectorized so it takes and returns arrays.

    Args:
        ang: angle in Rz gate.

    Returns:
        Clifford angle: closest clifford angle.
    """
    ang = ang % (2*np.pi)
    ang_scaled = ang/(np.pi/2)
    # if just one min value, return the corresponding nearest cliff.
    if (abs((ang_scaled/0.5) - 1) > 10**(-6) and abs((
        ang_scaled/0.5) - 3) > 10**(-6) and (abs((ang_scaled/0.5) - 5)
                                             > 10**(-6))):
        index = int(np.round(ang_scaled)) % 4
        return CLIFFORD_ANGLES[index]
    # if two min values (ie two cliff gates equidistant) randomly choose the
    # cliff gate to return.
    else:
        index_list = [ang_scaled - 0.5, ang_scaled + 0.5]
        index = int(choice(index_list))
        return CLIFFORD_ANGLES[index]


# vectorize so function can take array of angles.
_closest_clifford = np.vectorize(_closest_clifford)


def _random_clifford(
    ang: float
) -> float:
    """Function to take angle and return the random Clifford angle note the
       usage of this function is vectorized so it takes and returns arrays.

    Args:
        ang: angle in Rz gate.

    Returns:
        Clifford angle: closest clifford angle.
    """
    random_index = randint(0, 3)
    clifford_angle = CLIFFORD_ANGLES[random_index]
    return clifford_angle


# vectorize so function can take array:
_random_clifford = np.vectorize(_random_clifford)


def _angle_to_probabilities(
    angle: float,
    sigma: float
) -> float:
    """Function to return probability disribtuion based on distance from
       angles to Clifford gates.

    Args:
        angle: angle to form probability distribution.
    Returns:
        discrete value of probability distribution calucalted from
        Prob_project = exp(-(dist/sigma)^2) where dist = sum(dists) is the
        sum of distances from each Clifford gate.
    """
    angle = angle % (2*np.pi)
    S = np.array([[1, 0.0], [0.0, 1j]])
    Rz = np.array([[1, 0.0], [0.0, np.exp(angle*1j)]])
    dists = []
    for i in range(4):
        if i == 0:
            i = 4
        diff = np.linalg.norm(Rz - S ** (i))
        dists.append(np.exp(-(diff / sigma) ** 2))
    return sum(dists)


# vectorize so function can take array of angles.
_angle_to_probabilities = np.vectorize(_angle_to_probabilities)


def _probabilistic_angle_to_clifford(
    ang: float,
    sigma: float,
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
    ang = ang % (2*np.pi)
    S = np.array([[1, 0.0], [0.0, 1j]])
    Rz = np.array([[1, 0.0], [0.0, np.exp(ang*1j)]])
    dists = []
    for i in range(4):
        if i == 0:
            i = 4
        diff = np.linalg.norm(Rz - S ** (i))
        dists.append(np.exp(-(diff/sigma) ** 2))
    prob_gate = [i/sum(dists) for i in dists]
    cliff_ang = np.random.choice(
        CLIFFORD_ANGLES, 1, replace=False, p=prob_gate)
    return cliff_ang


# vectorize so function can take array of angles.
_probabilistic_angle_to_clifford = np.vectorize(
    _probabilistic_angle_to_clifford)
