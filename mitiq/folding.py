"""Functions for folding gates in valid mitiq circuits.

Public functions work for any circuit types supported by mitiq.
Private functions work only for iternal mitiq circuit representations.
"""
from copy import deepcopy
from typing import Any, Callable, Iterable, List, Optional, Tuple, Union

import numpy as np

from cirq import Circuit, InsertStrategy, inverse, ops
from mitiq import QPROGRAM, SUPPORTED_PROGRAM_TYPES


class UnsupportedCircuitError(Exception):
    pass


# Helper functions
def _is_measurement(op: ops.Operation) -> bool:
    """Returns true if the operation's gate is a measurement, else False.

    Args:
        op: Gate operation.
    """
    return isinstance(op.gate, ops.measurement_gate.MeasurementGate)


def _pop_measurements(
    circuit: Circuit,
) -> List[List[Union[int, ops.Operation]]]:
    """Removes all measurements from a circuit.

    Args:
        circuit: a quantum circuit as a :class:`cirq.Circuit` object.

    Returns:
        measurements: list
    """
    measurements = [
        list(m) for m in circuit.findall_operations(_is_measurement)
    ]
    circuit.batch_remove(measurements)
    return measurements


def _append_measurements(
    circuit: Circuit, measurements: List[Union[int, ops.Operation]]
) -> None:
    """Appends all measurements into the final moment of the circuit.

    Args:
        circuit: a quantum circuit as a :class:`cirq.Circuit`.
        measurements: measurements to perform.
    """
    for i in range(len(measurements)):
        measurements[i][0] = (
            len(circuit) + 1
        )  # Make sure the moment to insert into is the last in the circuit
    circuit.batch_insert(measurements)


# Conversions
def convert_to_mitiq(circuit: QPROGRAM) -> Tuple[Circuit, str]:
    """Converts any valid input circuit to a mitiq circuit.

    Args:
        circuit: Any quantum circuit object supported by mitiq.
                 See mitiq.SUPPORTED_PROGRAM_TYPES.

    Raises:
        UnsupportedCircuitError: If the input circuit is not supported.

    Returns:
        circuit: Mitiq circuit equivalent to input circuit.
        input_circuit_type: Type of input circuit represented by a string.
    """
    if "qiskit" in circuit.__module__:
        from mitiq.mitiq_qiskit.conversions import _from_qiskit
        input_circuit_type = "qiskit"
        mitiq_circuit = _from_qiskit(circuit)
    elif isinstance(circuit, Circuit):
        input_circuit_type = "cirq"
        mitiq_circuit = circuit
    else:
        raise UnsupportedCircuitError(
            f"Circuit from module {circuit.__module__} is not supported.\n\n" +
            f"Circuit types supported by mitiq are \n{SUPPORTED_PROGRAM_TYPES}"
        )
    return mitiq_circuit, input_circuit_type


def convert_from_mitiq(circuit: Circuit, conversion_type: str) -> QPROGRAM:
    """Converts a mitiq circuit to a type specificed by the conversion type.

    Args:
        circuit: Mitiq circuit to convert.
        conversion_type: String specifier for the converted circuit type.
    """
    if conversion_type == "qiskit":
        from mitiq.mitiq_qiskit.conversions import _to_qiskit
        converted_circuit = _to_qiskit(circuit)
    elif isinstance(circuit, Circuit):
        converted_circuit = circuit
    else:
        raise UnsupportedCircuitError(
            f"Conversion to circuit of type {conversion_type} is not supported."
            f"\nCircuit types supported by mitiq are {SUPPORTED_PROGRAM_TYPES}"
        )
    return converted_circuit


def converter(fold_method: Callable) -> Callable:
    """Decorator for handling conversions."""
    def new_fold_method(circuit: QPROGRAM, *args, **kwargs) -> QPROGRAM:
        mitiq_circuit, input_circuit_type = convert_to_mitiq(circuit)
        if kwargs.get("keep_input_type"):
            return convert_from_mitiq(
                fold_method(mitiq_circuit, *args, **kwargs), input_circuit_type
            )
        return fold_method(mitiq_circuit, *args, **kwargs)
    return new_fold_method


# Gate level folding
def _fold_gate_at_index_in_moment(
    circuit: Circuit, moment_index: int, gate_index: int
) -> None:
    """Replaces, in a circuit, the gate G in (moment, index) with G G^dagger G.

    Args:
        circuit: Circuit to fold.
        moment_index: Moment in which the gate sits in the circuit.
        gate_index: Index of the gate within the specified moment.
    """
    op = circuit[moment_index].operations[gate_index]
    circuit.insert(
        moment_index, [op, inverse(op)], strategy=InsertStrategy.NEW
    )


def _fold_gates_in_moment(
    circuit: Circuit, moment_index: int, gate_indices: Iterable[int]
) -> None:
    """Modifies the input circuit by applying the map G -> G G^dag G to all
    gates specified by the input moment index and gate indices.

     Args:
         circuit: Circuit to fold.
         moment_index: Index of moment to fold gates in.
         gate_indices: Indices of gates within the moments to fold.
     """
    for (i, gate_index) in enumerate(gate_indices):
        _fold_gate_at_index_in_moment(
            circuit, moment_index + 2 * i, gate_index
        )  # Each fold adds two moments


@converter
def fold_gates(
    circuit: QPROGRAM,
    moment_indices: Iterable[int],
    gate_indices: List[Iterable[int]],
    **kwargs,
) -> QPROGRAM:
    """Returns a new circuit with specified gates folded.

    Args:
        circuit: Circuit to fold.
        moment_indices: Indices of moments with gates to be folded.
        gate_indices: Specifies which gates within each moment to fold.

    Keyword Args:
        keep_input_type: If True, returns a circuit of the input type, else
                         returns a mitiq circuit.

    Returns:
        folded: the folded quantum circuit as a :class:`cirq.Circuit` object.

    Examples:
        (1) Folds the first three gates in moment two.
        >>> fold_gates(circuit, moment_indices=[1], gate_indices=[(0, 1, 2)])

        (2) Folds gates with indices 1, 4, and 5 in moment 0,
            and gates with indices 0, 1, and 2 in moment 1.
        >>> fold_gates(circuit, moment_indices=[0, 3],
        >>>                                gate_indices=[(1, 4, 5), (0, 1, 2)])
    """
    folded = deepcopy(circuit)
    moment_index_shift = 0
    for (i, moment_index) in enumerate(moment_indices):
        _fold_gates_in_moment(
            folded, moment_index + moment_index_shift, gate_indices[i]
        )
        moment_index_shift += 2 * len(
            gate_indices[i]
        )  # Folding gates adds moments
    return folded


def _fold_moments(circuit: Circuit, moment_indices: List[int]) -> None:
    """Folds specified moments in the circuit in place.

    Args:
        circuit: Circuit to fold.
        moment_indices: Indices of moments to fold in the circuit.

    """
    shift = 0
    for i in moment_indices:
        circuit.insert(
            i + shift, [circuit[i + shift], inverse(circuit[i + shift])]
        )
        shift += 2


@converter
def fold_moments(circuit: QPROGRAM,
                 moment_indices: List[int],
                 **kwargs
                 ) -> QPROGRAM:
    """Returns a new circuit with moments folded by mapping

    M_i -> M_i M_i^dag M_i

    where M_i is a moment specified by an integer in moment_indices.

    Args:
        circuit: Circuit to apply folding operation to.
        moment_indices: List of integers that specify moments to fold.

    Keyword Args:
        keep_input_type: If True, returns a circuit of the input type, else
                         returns a mitiq circuit.

    Returns:
        folded: the folded quantum circuit as a :class:`cirq.Circuit` object.
    """
    folded = deepcopy(circuit)
    _fold_moments(folded, moment_indices)
    return folded


def _fold_all_gates_locally(circuit: Circuit) -> None:
    """Replaces every gate G with G G^dag G by modifying the circuit in place.
    """
    _fold_moments(circuit, list(range(len(circuit))))


def _get_num_to_fold(stretch: float, ngates: int) -> int:
    """Returns the number of gates to fold to achieve the desired (approximate)
    stretch factor.

    Args:
        stretch: Floating point value to stretch the circuit by.
        ngates: Number of gates in the circuit to stretch.
    """
    return int(round(ngates * (stretch - 1.0) / 2.0))


@converter
def fold_gates_from_left(
        circuit: QPROGRAM, stretch: float, **kwargs
) -> QPROGRAM:
    """Returns a new folded circuit by applying the map G -> G G^dag G to a
    subset of gates of the input circuit, starting with gates at the
    left (beginning) of the circuit.

    The folded circuit has a number of gates approximately equal to
    stretch * n where n is the number of gates in the input circuit.

    Args:
        circuit: Circuit to fold.
        stretch: Factor to stretch the circuit by. Any real number in [1, 3].

    Keyword Args:
        keep_input_type: If True, returns a circuit of the input type, else
                         returns a mitiq circuit.

    Returns:
        folded: the folded quantum circuit as a :class:`cirq.Circuit` object.

    Note:
        Folding a single gate adds two gates to the circuit,
        hence the maximum stretch factor is 3.
    """
    if not circuit.are_all_measurements_terminal():
        raise ValueError(
            f"Input circuit contains intermediate measurements"
            " and cannot be folded."
        )

    if not 1 <= stretch <= 3:
        raise ValueError(
            "The stretch factor must be a real number between 1 and 3."
        )

    folded = deepcopy(circuit)

    measurements = _pop_measurements(folded)

    ngates = len(list(folded.all_operations()))
    num_to_fold = _get_num_to_fold(stretch, ngates)
    if num_to_fold == 0:
        _append_measurements(folded, measurements)
        return folded
    num_folded = 0
    moment_shift = 0

    for (moment_index, moment) in enumerate(circuit):
        for gate_index in range(len(moment)):
            _fold_gate_at_index_in_moment(
                folded, moment_index + moment_shift, gate_index
            )
            moment_shift += 2
            num_folded += 1
            if num_folded == num_to_fold:
                _append_measurements(folded, measurements)
                return folded


@converter
def fold_gates_from_right(
        circuit: QPROGRAM, stretch: float, **kwargs
) -> Circuit:
    """Returns a new folded circuit by applying the map G -> G G^dag G
    to a subset of gates of the input circuit, starting with gates at
    the right (end) of the circuit.

    The folded circuit has a number of gates approximately equal to
    stretch * n where n is the number of gates in the input circuit.

    Args:
        circuit: Circuit to fold.
        stretch: Factor to stretch the circuit by. Any real number in [1, 3].

    Keyword Args:
        keep_input_type: If True, returns a circuit of the input type, else
                         returns a mitiq circuit.

    Returns:
        folded: the folded quantum circuit as a :class:`cirq.Circuit` object.

    Note:
        Folding a single gate adds two gates to the circuit,
        hence the maximum stretch factor is 3.
    """
    if not circuit.are_all_measurements_terminal():
        raise ValueError(
            f"Input circuit contains intermediate measurements" \
            " and cannot be folded."
        )

    measurements = _pop_measurements(circuit)

    reversed_circuit = Circuit(reversed(circuit))
    reversed_folded_circuit = fold_gates_from_left(reversed_circuit, stretch)
    folded = Circuit(reversed(reversed_folded_circuit))
    _append_measurements(folded, measurements)
    return folded


def _update_moment_indices(
    moment_indices: dict, moment_index_where_gate_was_folded: int
) -> dict:
    """Updates moment indices to keep track of an original circuit
    throughout folding.

    Args:
        moment_indices: A dictionary in the format
                        {index of moment in original circuit: index of moment
                        in folded circuit}

        moment_index_where_gate_was_folded: Index of the moment
        in which a gate was folded.

    Returns:
        moment_indices: dictionary with updated moments.

    Note:
        `moment_indices` should start out as
        {0: 0, 1: 1, ..., M - 1: M - 1} where M is the # of moments in the
        original circuit. As the circuit is folded, moment indices change.

        If a gate in the last moment is folded, moment_indices gets updates to
        {0: 0, 1: 1, ..., M - 1:, M + 1} since two moments are created in the
        process of folding the gate in the last moment.

    TODO:
        If another gate from the last moment is folded, we could put it
        in the same moment as the previous folded gate.
    """
    if moment_index_where_gate_was_folded not in moment_indices.keys():
        raise ValueError(
            f"Moment index {moment_index_where_gate_was_folded} not in moment"\
            " indices"
        )
    for i in moment_indices.keys():
        moment_indices[i] += 2 * int(i >= moment_index_where_gate_was_folded)
    return moment_indices


@converter
def fold_gates_at_random(
    circuit: QPROGRAM, stretch: float, seed: Optional[int] = None, **kwargs
) -> QPROGRAM:
    """Returns a folded circuit by applying the map G -> G G^dag G to a random
    subset of gates in the input circuit.

    The folded circuit has a number of gates approximately equal to
     stretch * n where n is the number of gates in the input circuit.

    Args:
        circuit: Circuit to fold.
        stretch: Factor to stretch the circuit by. Any real number in [1, 3].
        seed: [Optional] Integer seed for random number generator.

    Keyword Args:
        keep_input_type: If True, returns a circuit of the input type, else
                         returns a mitiq circuit.

    Returns:
        folded: The folded quantum circuit as a :class:`cirq.Circuit` object.

    Note:
        Folding a single gate adds two gates to the circuit, hence the maximum
        stretch factor is 3.
    """
    if not circuit.are_all_measurements_terminal():
        raise ValueError(
            f"Input circuit contains intermediate measurements"
            " and cannot be folded."
        )

    if not 1 <= stretch <= 3:
        raise ValueError(
            "The stretch factor must be a real number between 1 and 3."
        )

    folded = deepcopy(circuit)

    measurements = _pop_measurements(folded)

    if np.isclose(stretch, 3.0, atol=1e-3):
        _fold_all_gates_locally(folded)
        _append_measurements(folded, measurements)
        return folded

    if seed:
        np.random.seed(seed)

    ngates = len(list(folded.all_operations()))
    num_to_fold = _get_num_to_fold(stretch, ngates)

    # Keep track of where moments are in the folded circuit
    moment_indices = {i: i for i in range(len(circuit))}

    # Keep track of which gates we can fold in each moment
    remaining_gate_indices = {
        moment: list(range(len(circuit[moment])))
        for moment in range(len(circuit))
    }

    # Any moment with at least one gate is fair game
    remaining_moment_indices = [
        i for i in remaining_gate_indices.keys() if remaining_gate_indices[i]
    ]

    for _ in range(num_to_fold):
        # Get a moment index and gate index from the remaining set
        moment_index = np.random.choice(remaining_moment_indices)
        gate_index = np.random.choice(remaining_gate_indices[moment_index])

        # Do the fold
        _fold_gate_at_index_in_moment(
            folded, moment_indices[moment_index], gate_index
        )

        # Update the moment indices for the folded circuit
        _update_moment_indices(moment_indices, moment_index)

        # Remove the gate we folded from the remaining set of gates to fold
        remaining_gate_indices[moment_index].remove(gate_index)

        # If there are no gates left in the moment,
        # remove the moment index from the remaining set
        if not remaining_gate_indices[moment_index]:
            remaining_moment_indices.remove(moment_index)

    _append_measurements(folded, measurements)
    return folded


@converter
def fold_local(
    circuit: QPROGRAM,
    stretch: float,
    fold_method: Callable[
        [Circuit, float, Tuple[Any]], Circuit
    ] = fold_gates_from_left,
    fold_method_args: Tuple[Any] = (),
    **kwargs
) -> QPROGRAM:
    """Returns a folded circuit by folding gates according to the input
    fold method.

    Args:
        circuit: Circuit to fold.
        stretch: Factor to stretch the circuit by.
        fold_method: Function which defines the method for folding gates.
        fold_method_args: Any additional input arguments for the fold_method.
                          The method is called with
                          fold_method(circuit, stretch, *fold_method_args).

    Keyword Args:
        keep_input_type: If True, returns a circuit of the input type, else
                         returns a mitiq circuit.

    Returns:
        folded: The folded quantum circuit as a :class:`cirq.Circuit` object.

    Example:
        >>> fold_method = fold_gates_at_random
        >>> fold_method_args = (1,)
        Uses a seed of one for the fold_gates_at_random method.

    Note:
        `fold_method` defines the strategy for folding gates, which could be
        folding gates at random, from the left of the circuit,
        or custom strategies.

        The signature of `fold_method` must be
            ```
            def fold_method(circuit: Circuit, stretch: float,**kwargs):
                ...
            ```
        and return a circuit.
    """
    folded = deepcopy(circuit)

    if np.isclose(stretch, 1.0, atol=1e-2):
        return folded

    if not 1 <= stretch:
        raise ValueError(
            f"The stretch factor must be a real number greater than 1."
        )

    while stretch > 1.0:
        this_stretch = 3.0 if stretch > 3.0 else stretch
        folded = fold_method(folded, this_stretch, *fold_method_args)
        stretch /= 3.0
    return folded


# Circuit level folding
@converter
def fold_global(circuit: QPROGRAM, stretch: float, **kwargs) -> QPROGRAM:
    """Gives a circuit by folding the global unitary of the input circuit.

    The returned folded circuit has a number of gates approximately equal to
     stretch * len(circuit).

    Args:
        circuit: Circuit to fold.
        stretch: Factor to stretch the circuit by.

    Keyword Args:
        keep_input_type: If True, returns a circuit of the input type, else
                         returns a mitiq circuit.

    Returns:
        folded: The folded quantum circuit as a :class:`cirq.Circuit` object.
    """
    if not (stretch >= 1):
        raise ValueError("The stretch factor must be a real number >= 1.")

    if not circuit.are_all_measurements_terminal():
        raise ValueError(
            "Input circuit contains intermediate measurements"
            " and cannot be folded."
        )

    folded = deepcopy(circuit)
    measurements = _pop_measurements(folded)
    base_circuit = deepcopy(folded)

    # Determine the number of global folds and the final fractional stretch
    num_global_folds, fractional_stretch = divmod(stretch - 1, 2)
    # Do the global folds
    for _ in range(int(num_global_folds)):
        folded += Circuit(inverse(base_circuit), base_circuit)

    # Fold remaining gates until the stretch is reached
    ops = list(base_circuit.all_operations())
    num_to_fold = int(round(fractional_stretch * len(ops) / 2))

    if num_to_fold > 0:
        folded += Circuit([inverse(ops[-num_to_fold:])], [ops[-num_to_fold:]])

    _append_measurements(folded, measurements)
    return folded
