"""Functions to fold gates in Cirq circuits."""
from copy import deepcopy
from typing import (Callable, Iterable, List)

import numpy as np

from cirq import (Circuit, InsertStrategy, inverse)


MAX_STRETCH_FACTOR = 100


# Gate level folding
def fold_gate_at_index_in_moment(circuit: Circuit, moment_index: int, gate_index: int) -> Circuit:
    """Returns a new circuit where the gate G in (moment, index) is replaced by G G^dagger G.

    Args:
        circuit: Circuit to fold.
        moment_index: Moment in which the gate sits in the circuit.
        gate_index: Index of the gate within the specified moment.
    """
    folded = deepcopy(circuit)
    op = folded[moment_index].operations[gate_index]
    folded.insert(moment_index, [inverse(op), op], strategy=InsertStrategy.NEW)
    return folded


def fold_gates_in_moment(circuit: Circuit, moment_index: int, gate_indices: Iterable[int]) -> Circuit:
    """Returns a new circuit which applies the map G -> G G^dag G to all gates specified by
     the input moment index and gate indices.

     Args:
         circuit: Circuit to fold.
         moment_index: Index of moment to fold gates in.
         gate_indices: Indices of gates within the moments to fold.
     """
    folded = deepcopy(circuit)
    for (i, gate_index) in enumerate(gate_indices):
        folded = fold_gate_at_index_in_moment(folded, moment_index + 2 * i, gate_index)  # Each fold adds two moments
    return folded


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
        folded = fold_gates_in_moment(folded, moment_index + moment_index_shift, gate_indices[i])
        moment_index_shift += 2 * len(gate_indices[i])  # Folding gates adds moments
    return folded


def fold_moments(circuit: Circuit, moment_indices: List[int]) -> Circuit:
    """Returns a new circuit with moments folded by mapping

    M_i -> M_i M_i^dag M_i

    where M_i is a moment specified by an integer in moment_indices.

    Args:
        circuit: Circuit to apply folding operation to.
        moment_indices: List of integers that specify moments to fold.
    """
    folded = deepcopy(circuit)
    shift = 0
    for i in moment_indices:
        folded.insert(i + shift, [inverse(circuit[i]), circuit[i]])
        shift += 2
    return folded


def fold_gates_from_left(circuit: Circuit, stretch: float) -> Circuit:
    """Returns a new folded circuit by applying the map G -> G G^dag G to a subset of gates  of the input circuit.

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
    num_to_fold = int(ngates * (stretch - 1.0) / 2.0)
    num_folded = 0
    moment_shift = 0

    for (moment_index, moment) in enumerate(circuit):
        for gate_index in range(len(moment)):
            # TODO: It could be expensive to call fold_gate...(...) which makes a deepcopy of the circuit each call.
            #  ==> Possible fix: Have fold_gate_at_index_in_moment(...) modify the circuit in place.
            folded = fold_gate_at_index_in_moment(folded, moment_index + moment_shift, gate_index)
            moment_shift += 2
            num_folded += 1
            if num_folded == num_to_fold:
                return folded


def fold_gates_at_random(circuit: Circuit, stretch: float, **kwargs) -> Circuit:
    """Returns a folded circuit by applying the map G -> G G^dag G to a random subset of gates in the input circuit.

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

    if "seed" in kwargs.keys():
        np.random.seed(kwargs.get("seed"))

    ngates = len(list(folded.all_operations()))
    num_to_fold = int(ngates * (stretch - 1.0) / 2.0)

    for _ in range(num_to_fold):
        # TODO: This allows for gates to be folded more than once, and folded gates to be folded.
        #  Should this be allowed?
        moment_index = np.random.choice(len(folded))
        gate_index = np.random.choice(len(folded[moment_index]))
        # TODO: It could be expensive to call fold_gate...(...) which makes a deepcopy of the circuit each call.
        #  ==> Possible fix: Have fold_gate_at_index_in_moment(...) modify the circuit in place.
        folded = fold_gate_at_index_in_moment(folded, moment_index, gate_index)

    return folded


def fold_local(circuit: Circuit, stretch: float, fold_method: Callable = fold_gates_from_left, **kwargs) -> Circuit:
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
    """
    folded = deepcopy(circuit)

    if np.isclose(stretch, 1.0, atol=1e-2):
        return folded

    if not 1 < stretch <= MAX_STRETCH_FACTOR:
        raise ValueError(f"The stretch factor must be a real number between 1 and {MAX_STRETCH_FACTOR}.")

    while stretch > 1.:
        this_stretch = 3. if stretch > 3. else stretch
        # TODO: This also allows folding gates that have already been folded. Should this be allowed?
        folded = fold_method(folded, this_stretch, **kwargs)
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
