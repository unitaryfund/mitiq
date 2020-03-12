"""Functions to fold gates in Cirq circuits."""
from copy import deepcopy
from typing import (Any, Callable, Iterable, List, Optional, Tuple)

import numpy as np

from cirq import (Circuit, InsertStrategy, inverse)


# Gate level folding
def _fold_gate_at_index_in_moment(circuit: Circuit, moment_index: int, gate_index: int) -> None:
    """Modifies the input circuit by replacing the gate G in (moment, index) is replaced by G G^dagger G.

    Args:
        circuit: Circuit to fold.
        moment_index: Moment in which the gate sits in the circuit.
        gate_index: Index of the gate within the specified moment.

    Returns:
        None
    """
    op = circuit[moment_index].operations[gate_index]
    circuit.insert(moment_index, [op, inverse(op)], strategy=InsertStrategy.NEW)


def _fold_gates_in_moment(circuit: Circuit, moment_index: int, gate_indices: Iterable[int]) -> None:
    """Modifies the input circuit by applying the map G -> G G^dag G to all gates specified by
     the input moment index and gate indices.

     Args:
         circuit: Circuit to fold.
         moment_index: Index of moment to fold gates in.
         gate_indices: Indices of gates within the moments to fold.

     Returns:
         None
     """
    for (i, gate_index) in enumerate(gate_indices):
        _fold_gate_at_index_in_moment(circuit, moment_index + 2 * i, gate_index)  # Each fold adds two moments


def fold_gates(circuit: Circuit, moment_indices: Iterable[int], gate_indices: List[Iterable[int]]) -> Circuit:
    """Returns a new circuit with specified gates folded.

    Args:
        circuit: Circuit to fold.
        moment_indices: Indices of moments with gates to be folded.
        gate_indices: Specifies which gates within each moment to fold.

    Examples:
        (1) Folds the first three gates in moment two.
        >>> fold_gates(circuit, moment_indices=[1], gate_indices=[(0, 1, 2)])

        (2) Folds gates with indices 1, 4, and 5 in moment 0,
            and gates with indices 0, 1, and 2 in moment 1.
        >>> fold_gates(circuit, moment_indices=[0, 3], gate_indices=[(1, 4, 5), (0, 1, 2)])
    """
    folded = deepcopy(circuit)
    moment_index_shift = 0
    for (i, moment_index) in enumerate(moment_indices):
        _fold_gates_in_moment(folded, moment_index + moment_index_shift, gate_indices[i])
        moment_index_shift += 2 * len(gate_indices[i])  # Folding gates adds moments
    return folded


def _fold_moments(circuit: Circuit, moment_indices: List[int]) -> None:
    """Folds specified moments in the circuit in place.

    Args:
        circuit: Circuit to fold.
        moment_indices: Indices of moments to fold in the circuit.

    Returns:
        None
    """
    shift = 0
    for i in moment_indices:
        circuit.insert(i + shift, [circuit[i + shift], inverse(circuit[i + shift])])
        shift += 2


def fold_moments(circuit: Circuit, moment_indices: List[int]) -> Circuit:
    """Returns a new circuit with moments folded by mapping

    M_i -> M_i M_i^dag M_i

    where M_i is a moment specified by an integer in moment_indices.

    Args:
        circuit: Circuit to apply folding operation to.
        moment_indices: List of integers that specify moments to fold.
    """
    folded = deepcopy(circuit)
    _fold_moments(folded, moment_indices)
    return folded


def _fold_all_gates_locally(circuit: Circuit) -> Circuit:
    """Replaces every gate G with G G^dag G by modifying the circuit in place."""
    _fold_moments(circuit, list(range(len(circuit))))


def _get_num_to_fold(stretch: float, ngates: int) -> int:
    """Returns the number of gates to fold to acheive the desired (approximate) stretch factor.

    Args:
        stretch: Floating point value to stretch the circuit by. Between 1 and 3.
        ngates: Number of gates in the circuit to stretch.
    """
    return round(ngates * (stretch - 1.0) / 2.0)


def fold_gates_from_left(circuit: Circuit, stretch: float) -> Circuit:
    """Returns a new folded circuit by applying the map G -> G G^dag G to a subset of gates of the input circuit,
    starting with gates at the left (beginning) of the circuit.

    The folded circuit has a number of gates approximately equal to stretch * n where n is the number of gates in
    the input circuit.

    Args:
        circuit: Circuit to fold.
        stretch: Factor to stretch the circuit by. Any real number in the interval [1, 3].

    Note:
        Folding a single gate adds two gates to the circuit, hence the maximum stretch factor is 3.
    """
    folded = deepcopy(circuit)

    if np.isclose(stretch, 1.0, atol=1e-2):
        return folded

    if not 1 < stretch <= 3:
        raise ValueError("The stretch factor must be a real number between 1 and 3.")

    ngates = len(list(folded.all_operations()))
    num_to_fold = _get_num_to_fold(stretch, ngates)
    if num_to_fold == 0:
        return folded
    num_folded = 0
    moment_shift = 0

    for (moment_index, moment) in enumerate(circuit):
        for gate_index in range(len(moment)):
            _fold_gate_at_index_in_moment(folded, moment_index + moment_shift, gate_index)
            moment_shift += 2
            num_folded += 1
            if num_folded == num_to_fold:
                return folded


def fold_gates_from_right(circuit: Circuit, stretch: float) -> Circuit:
    """Returns a new folded circuit by applying the map G -> G G^dag G to a subset of gates of the input circuit,
    starting with gates at the right (end) of the circuit.

    The folded circuit has a number of gates approximately equal to stretch * n where n is the number of gates in
    the input circuit.

    Args:
        circuit: Circuit to fold.
        stretch: Factor to stretch the circuit by. Any real number in the interval [1, 3].

    Note:
        Folding a single gate adds two gates to the circuit, hence the maximum stretch factor is 3.
    """
    reversed_circuit = Circuit(reversed(circuit))
    reversed_folded_circuit = fold_gates_from_left(reversed_circuit, stretch)
    return Circuit(reversed(reversed_folded_circuit))


def _update_moment_indices(moment_indices: dict, moment_index_where_gate_was_folded: int) -> dict:
    """Updates moment indices to keep track of an original circuit throughout folding.

    Args:
        moment_indices: A dictionary in the format
                        {index of moment in original circuit: index of moment in folded circuit}

                        For example, moment_indices should start out as
                        {0: 0, 1: 1, ..., M - 1: M - 1}
                        where M is the number of moments in the original circuit.

                        As the circuit is folded, moment indices change. For example, if a gate in the last moment
                        is folded, moment_indices gets updates to
                        {0: 0, 1: 1, ..., M - 1:, M + 1}
                        since two moments are created in the process of folding the gate in the last moment.

                        TODO: If another gate from the last moment is folded, we could put it in the same moment as
                         the previous folded gate.

        moment_index_where_gate_was_folded: Index of the moment in which a gate was folded.
    """
    if moment_index_where_gate_was_folded not in moment_indices.keys():
        raise ValueError(f"Moment index {moment_index_where_gate_was_folded} not in moment indices")
    for i in moment_indices.keys():
        moment_indices[i] += 2 * int(i >= moment_index_where_gate_was_folded)
    return moment_indices


def fold_gates_at_random(circuit: Circuit, stretch: float, seed: Optional[int] = None) -> Circuit:
    """Returns a folded circuit by applying the map G -> G G^dag G to a random subset of gates in the input circuit.

    The folded circuit has a number of gates approximately equal to stretch * n where n is the number of gates in
    the input circuit.

    Args:
        circuit: Circuit to fold.
        stretch: Factor to stretch the circuit by. Any real number in the interval [1, 3].
        seed: [Optional] Integer seed for random number generator.

    Note:
        Folding a single gate adds two gates to the circuit, hence the maximum stretch factor is 3.
    """
    folded = deepcopy(circuit)

    if np.isclose(stretch, 1.0, atol=1e-3):
        return folded

    if np.isclose(stretch, 3.0, atol=1e-3):
        _fold_all_gates_locally(folded)
        return folded

    if not 1 < stretch <= 3:
        raise ValueError("The stretch factor must be a real number between 1 and 3.")

    if seed:
        np.random.seed(seed)

    ngates = len(list(folded.all_operations()))
    num_to_fold = _get_num_to_fold(stretch, ngates)

    # Keep track of where moments are in the folded circuit
    moment_indices = {i: i for i in range(len(circuit))}

    # Keep track of which gates we can fold in each moment
    remaining_gate_indices = {moment: list(range(len(circuit[moment]))) for moment in range(len(circuit))}

    # Any moment with at least one gate is fair game
    remaining_moment_indices = [i for i in remaining_gate_indices.keys() if remaining_gate_indices[i]]

    for _ in range(num_to_fold):
        # Get a moment index and gate index from the remaining set
        moment_index = np.random.choice(remaining_moment_indices)
        gate_index = np.random.choice(remaining_gate_indices[moment_index])

        # Do the fold
        _fold_gate_at_index_in_moment(folded, moment_indices[moment_index], gate_index)

        # Update the moment indices for the folded circuit
        _update_moment_indices(moment_indices, moment_index)

        # Remove the gate we folded from the remaining set of gates to fold
        remaining_gate_indices[moment_index].remove(gate_index)

        # If there are no gates left in the moment, remove the moment index from the remaining set
        if not remaining_gate_indices[moment_index]:
            remaining_moment_indices.remove(moment_index)

    return folded


def fold_local(
        circuit: Circuit,
        stretch: float,
        fold_method: Callable[[Circuit, float, Tuple[Any]], Circuit] = fold_gates_from_left,
        fold_method_args: Tuple[Any] = ()) -> Circuit:
    """Returns a folded circuit by folding gates according to the input fold method.

    Args:
        circuit: Circuit to fold.
        stretch: Factor to stretch the circuit by.
        fold_method: Function which defines the method for folding gates.
                    (e.g., Randomly selects gates to fold, folds gates starting from left of circuit, etc.)

                    Must have signature

                    def fold_method(circuit: Circuit, stretch: float, **kwargs):
                        ...

                    and return a circuit.
        fold_method_args: Any additional input arguments for the fold_method.
                          The method is called with fold_method(circuit, stretch, *fold_method_args).
            Example:
                fold_method = fold_gates_at_random
                fold_method_args = (1,)

                > Uses a seed of one for the fold_gates_at_random method.
    """
    folded = deepcopy(circuit)

    if np.isclose(stretch, 1.0, atol=1e-2):
        return folded

    if not 1 < stretch:
        raise ValueError(f"The stretch factor must be a real number greater than 1.")

    while stretch > 1.:
        this_stretch = 3. if stretch > 3. else stretch
        folded = fold_method(folded, this_stretch, *fold_method_args)
        stretch /= 3.
    return folded


# Circuit level folding
def unitary_folding(circuit: Circuit, stretch: float) -> Circuit:
    """Applies global unitary folding and a final partial folding of the input circuit.
    Returns a circuit of depth approximately equal to stretch*len(circuit).
    The stretch factor can be any real number >= 1."""

    if not (stretch >= 1):
        raise ValueError("The stretch factor must be a real number >= 1.")

    # determine the number of integer foldings and the final fractional_stretch
    num_foldings, fractional_stretch = divmod(stretch - 1, 2)

    # integer circuit folding
    eye = Circuit()
    for _ in range(int(num_foldings)):
        eye += inverse(circuit) + circuit

    # partial circuit folding.
    depth = len(circuit)
    fractional_depth = int(depth * fractional_stretch / 2)
    if fractional_depth != 0:
        eye += inverse(circuit[-fractional_depth:]) + circuit[-fractional_depth:]

    return circuit + eye