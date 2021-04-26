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

"""Functions for local and global unitary folding on supported circuits."""
from copy import deepcopy
from typing import (
    Any,
    Callable,
    Collection,
    Dict,
    FrozenSet,
    Iterable,
    List,
    Optional,
    Tuple,
    Union,
)

import numpy as np
from cirq import Circuit, InsertStrategy, inverse, ops, has_unitary

from mitiq._typing import QPROGRAM
from mitiq.conversions import noise_scaling_converter


class UnfoldableGateError(Exception):
    pass


class UnfoldableCircuitError(Exception):
    pass


_cirq_gates_to_string_keys = {
    ops.H: "H",
    ops.X: "X",
    ops.Y: "Y",
    ops.Z: "Z",
    ops.T: "T",
    ops.I: "I",
    ops.CNOT: "CNOT",
    ops.CZ: "CZ",
    ops.TOFFOLI: "TOFFOLI",
}
_string_keys_to_cirq_gates = dict(
    zip(_cirq_gates_to_string_keys.values(), _cirq_gates_to_string_keys.keys())
)


# Helper functions
def _is_measurement(op: ops.Operation) -> bool:
    """Returns true if the operation's gate is a measurement, else False.

    Args:
        op: Gate operation.
    """
    return isinstance(op.gate, ops.measurement_gate.MeasurementGate)


def _pop_measurements(circuit: Circuit,) -> List[Tuple[int, ops.Operation]]:
    """Removes all measurements from a circuit.

    Args:
        circuit: A quantum circuit as a :class:`cirq.Circuit` object.

    Returns:
        measurements: List of measurements in the circuit.
    """
    measurements = list(circuit.findall_operations(_is_measurement))
    circuit.batch_remove(measurements)
    return measurements


def _append_measurements(
    circuit: Circuit, measurements: List[Tuple[int, ops.Operation]]
) -> None:
    """Appends all measurements into the final moment of the circuit.

    Args:
        circuit: a quantum circuit as a :class:`cirq.Circuit`.
        measurements: measurements to perform.
    """
    new_measurements: List[Tuple[int, ops.Operation]] = []
    for i in range(len(measurements)):
        # Make sure the moment to insert into is the last in the circuit
        new_measurements.append((len(circuit) + 1, measurements[i][1]))
    circuit.batch_insert(new_measurements)


def _check_foldable(circuit: Circuit) -> None:
    """Raises an error if the input circuit cannot be folded.

    Args:
        circuit: Checks whether this circuit is able to be folded.

    Raises:
        UnfoldableCircuitError:
            * If the circuit has intermediate measurements.
            * If the circuit has non-unitary channels which are not terminal
              measurements.
    """
    if not circuit.are_all_measurements_terminal():
        raise UnfoldableCircuitError(
            "Circuit contains intermediate measurements and cannot be folded."
        )

    if not has_unitary(circuit):
        raise UnfoldableCircuitError(
            "Circuit contains non-unitary channels which are not terminal "
            "measurements and cannot be folded."
        )


def squash_moments(circuit: Circuit) -> Circuit:
    """Returns a copy of the input circuit with all gates squashed into as few
    moments as possible.

    Args:
        circuit: Circuit to squash moments of.
    """
    return Circuit(
        circuit.all_operations(),
        strategy=InsertStrategy.EARLIEST,
        device=circuit.device,
    )


# Gate level folding
def _fold_gate_at_index_in_moment(
    circuit: Circuit, moment_index: int, gate_index: int
) -> None:
    """Replaces the gate G at (moment, index) with G G^dagger G, modifying the
    circuit in place. Inserts two new moments into the circuit for G^dagger
    and G.

    Args:
        circuit: Circuit with gates to fold.
        moment_index: Moment in which the gate to be folded sits in the
            circuit.
        gate_index: Index of the gate to be folded within the specified
            moment.
    """
    moment = circuit[moment_index]
    op = moment.operations[gate_index]
    try:
        inverse_op = inverse(op)
    except TypeError:
        raise UnfoldableGateError(
            f"Operation {op} does not have a defined "
            f"inverse and cannot be folded."
        )
    circuit.insert(moment_index, [op, inverse_op], strategy=InsertStrategy.NEW)


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


def _fold_gates(
    circuit: Circuit,
    moment_indices: Iterable[int],
    gate_indices: List[Collection[int]],
) -> Circuit:
    """Returns a new circuit with specified gates folded.

    Args:
        circuit: Circuit to fold.
        moment_indices: Indices of moments with gates to be folded.
        gate_indices: Specifies which gates within each moment to fold.

    Returns:
        folded: the folded quantum circuit as a cirq.Circuit.

    Examples:
        (1) Folds the first three gates in moment two.
        >>> _fold_gates(circuit, moment_indices=[1], gate_indices=[(0, 1, 2)])

        (2) Folds gates with indices 1, 4, and 5 in moment 0,
            and gates with indices 0, 1, and 2 in moment 1.
        >>> _fold_gates(circuit,
        >>>             moment_indices=[0, 3],
        >>>             gate_indices=[(1, 4, 5), (0, 1, 2)])
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


def _fold_moments(
    circuit: Circuit, moment_indices: List[int], num_folds: int = 1
) -> None:
    """Folds specified moments in the circuit in place.

    Args:
        circuit: Circuit to fold.
        moment_indices: Indices of moments to fold in the circuit.
        num_folds: Number of times to fold each moment.
    """
    shift = 0
    for i in moment_indices:
        circuit.insert(
            i + shift,
            [circuit[i + shift], inverse(circuit[i + shift])] * num_folds,
        )
        shift += 2 * num_folds


def _fold_all(
    circuit: Circuit,
    num_folds: int = 1,
    exclude: FrozenSet[Any] = frozenset(),
    skip_moments: FrozenSet[int] = frozenset(),
) -> Circuit:
    """Returns a circuit with all gates folded locally.

    Args:
        circuit: Circuit to fold.
        num_folds: Number of times to add G G^dag for each gate G. If not an
            integer, this gets rounded to the nearest integer.
        exclude: Do not fold these gates.
        skip_moments: Do not fold these moments.
    """
    num_folds = round(num_folds)
    if num_folds < 0:
        raise ValueError(
            f"Arg `num_folds` must be positive but was {num_folds}."
        )

    # Parse the exclude argument.
    all_gates = set(op.gate for op in circuit.all_operations())
    to_exclude = set()
    for item in exclude:
        if isinstance(item, str):
            try:
                to_exclude.add(_string_keys_to_cirq_gates[item])
            except KeyError:
                if item == "single":
                    to_exclude.update(
                        gate for gate in all_gates if gate.num_qubits() == 1
                    )
                elif item == "double":
                    to_exclude.update(
                        gate for gate in all_gates if gate.num_qubits() == 2
                    )
                elif item == "triple":
                    to_exclude.update(
                        gate for gate in all_gates if gate.num_qubits() == 3
                    )
                else:
                    raise ValueError(
                        f"Do not know how to parse item '{item}' in exclude. "
                        f"Valid items are Cirq gates, string keys specifying"
                        f"gates, and 'single', 'double', or 'triple'."
                    )
        elif isinstance(item, ops.Gate):
            to_exclude.add(item)
        else:
            raise ValueError(
                f"Do not know how to exclude {item} of type {type(item)}."
            )

    folded = deepcopy(circuit)[:0]
    for (i, moment) in enumerate(circuit):
        if i in skip_moments:
            folded.append(moment, strategy=InsertStrategy.EARLIEST)
            continue

        for op in moment:
            folded.append(op, strategy=InsertStrategy.EARLIEST)
            if op.gate not in to_exclude:
                folded.append(
                    [inverse(op), op] * num_folds,
                    strategy=InsertStrategy.EARLIEST,
                )

    return folded


# Helper functions for folding by fidelity
def _default_weight(op: ops.Operation) -> float:
    """Returns a default weight for an operation."""
    return 0.99 ** len(op.qubits)


def _get_weight_for_gate(
    weights: Union[Dict[str, float], None], op: ops.Operation
) -> float:
    """Returns the weight for a given gate, using a default value of 1.0 if
    weights is None or if the weight is not specified.

    Args:
        weights: Dictionary of string keys mapping gates to weights.
        op: Operation to get the weight of.
    """
    weight = _default_weight(op)
    if not weights:
        return weight

    if "single" in weights.keys() and len(op.qubits) == 1:
        weight = weights["single"]
    elif "double" in weights.keys() and len(op.qubits) == 2:
        weight = weights["double"]
    elif "triple" in weights.keys() and len(op.qubits) == 3:
        weight = weights["triple"]

    if op.gate and op.gate in _cirq_gates_to_string_keys.keys():
        # Get the string key for this gate
        key = _cirq_gates_to_string_keys[op.gate]
        if key in weights.keys():
            weight = weights[_cirq_gates_to_string_keys[op.gate]]
    return weight


def _compute_weight(circuit: Circuit, weights: Dict[str, float]) -> float:
    """Returns the weight of the circuit as the sum of weights of individual
    gates. Gates not defined have a default weight of one.

    Args:
        circuit: Circuit to compute the weight of.
        weights: Dictionary mapping string keys of gates to weights.
    """
    return sum(
        _get_weight_for_gate(weights, op) for op in circuit.all_operations()
    )


def _get_num_to_fold(scale_factor: float, ngates: int) -> int:
    """Returns the number of gates to fold to achieve the desired
    (approximate) scale factor.

    Args:
        scale_factor: Floating point value to scale the circuit by.
        ngates: Number of gates in the circuit to fold.
    """
    return int(round(ngates * (scale_factor - 1.0) / 2.0))


# Local folding functions
@noise_scaling_converter
def fold_all(
    circuit: QPROGRAM,
    scale_factor: float,
    exclude: FrozenSet[Any] = frozenset(),
) -> QPROGRAM:
    """Returns a circuit with all gates folded locally.

    Args:
        circuit: Circuit to fold.
        scale_factor: Approximate factor by which noise is scaled in the
            circuit. Each gate is folded round((scale_factor - 1.0) / 2.0)
            times. For example::

                scale_factor | num_folds
                ------------------------
                1.0          | 0
                3.0          | 1
                5.0          | 2

        exclude: Do not fold these gates. Supported gate keys are listed in
            the following table.::

                Gate key    | Gate
                -------------------------
                "H"         | Hadamard
                "X"         | Pauli X
                "Y"         | Pauli Y
                "Z"         | Pauli Z
                "I"         | Identity
                "CNOT"      | CNOT
                "CZ"        | CZ gate
                "TOFFOLI"   | Toffoli gate
                "single"    | All single qubit gates
                "double"    | All two-qubit gates
                "triple"    | All three-qubit gates
    """
    if not 1.0 <= scale_factor:
        raise ValueError(
            f"Requires scale_factor >= 1 but scale_factor = {scale_factor}."
        )
    _check_foldable(circuit)

    folded = deepcopy(circuit)
    measurements = _pop_measurements(folded)

    folded = _fold_all(folded, round((scale_factor - 1.0) / 2.0), exclude)

    _append_measurements(folded, measurements)
    return folded


def _update_moment_indices(
    moment_indices: Dict[int, int], moment_index_where_gate_was_folded: int
) -> Dict[int, int]:
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
    """
    if moment_index_where_gate_was_folded not in moment_indices.keys():
        raise ValueError(
            f"Moment index {moment_index_where_gate_was_folded} not in moment"
            " indices."
        )
    for i in moment_indices.keys():
        moment_indices[i] += 2 * int(i >= moment_index_where_gate_was_folded)
    return moment_indices


def _fold_local(
    circuit: Circuit,
    scale_factor: float,
    fold_method: Callable[..., Circuit],
    fold_method_args: Optional[Tuple[Any]] = None,
    **kwargs: Any,
) -> Circuit:
    """Helper function for implementing a local folding method (which nominally
    requires 1 <= scale_factor <= 3) at any scale_factor >= 1. Returns a folded
    circuit by folding gates according to the input fold method.

    Args:
        circuit: Circuit to fold.
        scale_factor: Factor to scale the circuit by.
        fold_method: Function which defines the method for folding gates.
        fold_method_args: Any additional input arguments for the fold_method.
            The method is called with
            fold_method(circuit, scale_factor, *fold_method_args).

    Keyword Args:
        squash_moments (bool): If True, all gates (including folded gates) are
            placed as early as possible in the circuit. If False, new moments
            are created for folded gates. This option only applies to QPROGRAM
            types which have a "moment" or "time" structure. Default is True.

    Returns:
        folded: The folded quantum circuit as a Cirq Circuit.

    Note:
        `fold_method` defines the strategy for folding gates, which could be
        folding gates at random, from the left of the circuit,
        or custom strategies.

        The signature of `fold_method` must be
            ```
            def fold_method(circuit: Circuit, scale_factor: float,**kwargs):
                ...
            ```
        and return a circuit.
    """
    folded = deepcopy(circuit)

    if np.isclose(scale_factor, 1.0, atol=1e-2):
        return folded

    if not 1 <= scale_factor:
        raise ValueError(
            "The scale factor must be a real number greater than 1."
        )

    while scale_factor > 1.0:
        this_stretch = 3.0 if scale_factor > 3.0 else scale_factor
        if fold_method_args:
            folded = fold_method(
                folded,
                this_stretch,
                *fold_method_args,
                squash_moments=False,
                **{k: v for k, v in kwargs.items() if k != "squash_moments"},
            )
        else:
            folded = fold_method(
                folded,
                this_stretch,
                squash_moments=False,
                **{k: v for k, v in kwargs.items() if k != "squash_moments"},
            )
        scale_factor /= 3.0

    if not (kwargs.get("squash_moments") is False):
        folded = squash_moments(folded)
    return folded


# Global folding function
@noise_scaling_converter
def fold_global(
    circuit: QPROGRAM, scale_factor: float, **kwargs: Any
) -> QPROGRAM:
    """Returns a new circuit obtained by folding the global unitary of the
    input circuit.

    The returned folded circuit has a number of gates approximately equal to
    scale_factor * len(circuit).

    Args:
        circuit: Circuit to fold.
        scale_factor: Factor to scale the circuit by.

    Keyword Args:
        squash_moments (bool): If True, all gates (including folded gates) are
            placed as early as possible in the circuit. If False, new moments
            are created for folded gates. This option only applies to QPROGRAM
            types which have a "moment" or "time" structure. Default is True.
        return_mitiq (bool): If True, returns a mitiq circuit instead of
            the input circuit type (if different). Default is False.

    Returns:
        folded: the folded quantum circuit as a QPROGRAM.
    """
    _check_foldable(circuit)

    if not (scale_factor >= 1):
        raise ValueError("The scale factor must be a real number >= 1.")

    folded = deepcopy(circuit)
    measurements = _pop_measurements(folded)
    base_circuit = deepcopy(folded)

    # Determine the number of global folds and the final fractional scale
    num_global_folds, fraction_scale = divmod(scale_factor - 1, 2)
    # Do the global folds
    for _ in range(int(num_global_folds)):
        folded += Circuit(inverse(base_circuit), base_circuit)

    # Fold remaining gates until the scale is reached
    operations = list(base_circuit.all_operations())
    num_to_fold = int(round(fraction_scale * len(operations) / 2))

    if num_to_fold > 0:
        folded += Circuit(
            [inverse(operations[-num_to_fold:])], [operations[-num_to_fold:]]
        )

    _append_measurements(folded, measurements)
    if not (kwargs.get("squash_moments") is False):
        folded = squash_moments(folded)
    return folded


def _create_weight_mask(
    circuit: Circuit, fidelities: Optional[Dict[str, float]],
) -> List[float]:
    """Returns a list of weights associated to each gate if the input
    circuit. Measurement gates are ignored.

    Args:
        circuit: The circuit from which a weight mask is created.
        fidelities: The dictionary of gate fidelities.
            If None, default fidelities will be used. See the
            docstring of local folding function for mode details.

    Returns: The list of weights associated to all the gates.
    """
    if fidelities and not all(0.0 < f <= 1.0 for f in fidelities.values()):
        raise ValueError("Fidelities should be in the interval (0, 1].")
    weights = None
    if fidelities:
        # Round to avoid ugly numbers like 0.09999999999999998 instead of 0.1
        # https://docs.python.org/2/tutorial/floatingpoint.html
        weights = {k: round(1.0 - f, 12) for k, f in fidelities.items()}

    # Build mask with weights of each gate
    return [
        _get_weight_for_gate(weights, op)
        for op in circuit.all_operations()
        if not _is_measurement(op)
    ]


def _create_fold_mask(
    weight_mask: List[float],
    scale_factor: float,
    folding_method: str,
    seed: Optional[int] = None,
) -> List[float]:
    r"""Returns a list of integers determining how many times each gate a
    circuit should be folded to realize the desired input scale_factor.

    More precicely, the j_th element of the returned list is associated
    to the j_th gate G_j of a circuit that we want to scale and determines
    how many times G_j^\dag G_j should be applied after G_j.

    The returned list is built such that the total weight of the
    folded circuit is approximately equal to scale_factor times the
    total weight of the input circuit.

    For equal weights, this function reproduces the local unitary folding
    method defined in equation (5) of [Giurgica_Tiron_2020_arXiv]_.

    Args:
        weight_mask: The weights of all the gates of the circuit to fold.
            Highly noisy gates should have a corresponding high weight.
            Gates with zero weight are assumed to be ideal and are not folded.
        scale_factor: The effective noise scale factor.
        folding_method: A string equal to "at_random", or "from_left", or
            "from_right". Determines the partial folding method described in
            [Giurgica_Tiron_2020_arXiv]_. If scale_factor is an odd integer,
            all methods are equivalent and this option is irrelevant.
        seed: A seed for the random number generator. This is used only when
            folding_method is "at_random".

    Returns: The list of integers determining to how many times one should
        fold the j_th gate of the circuit to be scaled.

    Example:
        >>>_create_fold_mask(
            weight_mask=[1.0, 0.5, 2.0, 0.0],
            scale_factor=4,
            folding_method="from_left",
        )
        [2, 2, 1, 0]
    """

    if not 1.0 <= scale_factor:
        raise ValueError(
            f"Requires scale_factor >= 1 but scale_factor = {scale_factor}."
        )

    # Find the maximum odd integer smaller or equal to scale_factor
    num_uniform_folds = int((scale_factor - 1.0) / 2.0)
    odd_integer_scale_factor = 2 * num_uniform_folds + 1

    # Uniformly folding all gates to reach odd_integer_scale_factor
    num_folds_mask = []
    for w in weight_mask:
        if np.isclose(w, 0.0):
            num_folds_mask.append(0)
        else:
            num_folds_mask.append(num_uniform_folds)

    # If the scale_factor is an odd integer, we are done.
    if np.isclose(odd_integer_scale_factor, scale_factor):

        return num_folds_mask

    # If necessary, fold a subset of gates to approximate the scale_factor
    input_circuit_weight = sum(weight_mask)
    output_circuit_weight = odd_integer_scale_factor * input_circuit_weight

    # Express folding order through a list of indices
    folding_order = list(range(len(weight_mask)))
    if folding_method == "from_left":
        pass
    elif folding_method == "from_right":
        folding_order.reverse()
    elif folding_method == "at_random":
        rnd_state = np.random.RandomState(seed)
        rnd_state.shuffle(folding_order)
    else:
        raise ValueError(
            "The option 'folding_method' is not valid."
            "It must be 'at_random', or 'from_left', or 'from_right'."
        )

    # Fold gates until the input scale_factor is better approximated
    approx_error = np.abs(
        output_circuit_weight - scale_factor * input_circuit_weight
    )
    for j in folding_order:
        # Skip gates with 0 weight
        if np.isclose(weight_mask[j], 0.0):
            continue

        # Compute the approx error if a new fold would be applied
        new_output_circuit_weight = output_circuit_weight + 2 * weight_mask[j]
        new_approx_error = np.abs(
            new_output_circuit_weight - scale_factor * input_circuit_weight
        )
        # Fold the candidate gate only if it helps improving the approximation
        if new_approx_error < approx_error:
            approx_error = new_approx_error
            output_circuit_weight = new_output_circuit_weight
            num_folds_mask[j] += 1
        # TODO: decide if removing next else case to get better approximations
        else:
            break

    return num_folds_mask


def _apply_fold_mask(
    circuit: Circuit,
    num_folds_mask: List[int],
    squash_moments: Optional[bool] = True,
) -> Circuit:
    r"""Applies local unitary folding to the gates of the input circuit
    according to the input num_folds_mask.

    More precicely, G_j^\dag G_j is applied after the j_th gate G_j of
    the input circuit an integer number of times given by num_folds_mask[j].

    Args:
        circuit: The quantum circuit to fold.
        num_folds_mask: The list of integers indicating how many times
            the corresponding gates of 'circuit' should be folded.
        squash_moments: If True or None, all gates (including folded gates) are
            placed as early as possible in the circuit. If False, new moments
            are created for folded gates. This option only applies to QPROGRAM
            types which have a "moment" or "time" structure. Default is True.

    Returns: The folded quantum circuit.
    """
    if squash_moments is False:
        pass
    else:
        squash_moments = True

    _check_foldable(circuit)
    circ_copy = deepcopy(circuit)
    measurements = _pop_measurements(circ_copy)

    num_gates = len(list(circ_copy.all_operations()))
    if num_gates != len(num_folds_mask):
        raise ValueError(
            "The circuit and the folding mask have incompatible sizes."
            f" The number of gates is {num_gates}"
            f" but len(num_folds_mask) is {len(num_folds_mask)}."
        )

    folded_circuit = circ_copy[:0]
    if squash_moments:
        for op, num_folds in zip(circ_copy.all_operations(), num_folds_mask):
            folded_circuit.append([op] + num_folds * [inverse(op), op])
    else:
        index = 0
        for moment in circ_copy:
            folded_moment = Circuit(moment)
            for op in moment:
                num_folds = num_folds_mask[index]
                folded_moment.append(num_folds * [inverse(op), op])
                index += 1
            # New folded gates are only squashed with respect to folded_moment
            # while folded_circuit is not squashed.
            folded_circuit.append(folded_moment)

    _append_measurements(folded_circuit, measurements)
    return folded_circuit


@noise_scaling_converter
def fold_gates_from_left(
    circuit: QPROGRAM, scale_factor: float, **kwargs: Any
) -> QPROGRAM:
    """Returns a new folded circuit by applying the map G -> G G^dag G to a
    subset of gates of the input circuit, starting with gates at the
    left (beginning) of the circuit.

    The folded circuit has a number of gates approximately equal to
    scale_factor * n where n is the number of gates in the input circuit.

    For equal gate fidelities, this function reproduces the local unitary
    folding method defined in equation (5) of [Giurgica_Tiron_2020_arXiv]_.

    Args:
        circuit: Circuit to fold.
        scale_factor: Factor to scale the circuit by. Any real number >= 1.

    Keyword Args:
        fidelities (Dict[str, float]): Dictionary of gate fidelities. Each key
            is a string which specifies the gate and each value is the
            fidelity of that gate. When this argument is provided, folded
            gates contribute an amount proportional to their infidelity
            (1 - fidelity) to the total noise scaling. Fidelity values must be
            in the interval (0, 1]. Gates not specified have a default
            fidelity of 0.99**n where n is the number of qubits the gates act
            on.

            Supported gate keys are listed in the following table.::

                Gate key    | Gate
                -------------------------
                "H"         | Hadamard
                "X"         | Pauli X
                "Y"         | Pauli Y
                "Z"         | Pauli Z
                "I"         | Identity
                "CNOT"      | CNOT
                "CZ"        | CZ gate
                "TOFFOLI"   | Toffoli gate
                "single"    | All single qubit gates
                "double"    | All two-qubit gates
                "triple"    | All three-qubit gates

            Keys for specific gates override values set by "single", "double",
            and "triple".

            For example, `fidelities = {"single": 1.0, "H", 0.99}` sets all
            single-qubit gates except Hadamard to have fidelity one.

        squash_moments (bool): If True, all gates (including folded gates) are
            placed as early as possible in the circuit. If False, new moments
            are created for folded gates. This option only applies to QPROGRAM
            types which have a "moment" or "time" structure. Default is True.

        return_mitiq (bool): If True, returns a mitiq circuit instead of
            the input circuit type (if different). Default is False.

    Returns:
        folded: The folded quantum circuit as a QPROGRAM.

    """

    weight_mask = _create_weight_mask(circuit, kwargs.get("fidelities"))

    num_folds_mask = _create_fold_mask(
        weight_mask, scale_factor, folding_method="from_left"
    )

    return _apply_fold_mask(
        circuit, num_folds_mask, squash_moments=kwargs.get("squash_moments"),
    )


@noise_scaling_converter
def fold_gates_from_right(
    circuit: QPROGRAM, scale_factor: float, **kwargs: Any
) -> QPROGRAM:
    r"""Returns a new folded circuit by applying the map G -> G G^dag G to a
    subset of gates of the input circuit, starting with gates at the
    right (end) of the circuit.

    The folded circuit has a number of gates approximately equal to
    scale_factor * n where n is the number of gates in the input circuit.

    For equal gate fidelities, this function reproduces the local unitary
    folding method defined in equation (5) of [Giurgica_Tiron_2020_arXiv]_.

    Args:
        circuit: Circuit to fold.
        scale_factor: Factor to scale the circuit by. Any real number >= 1.

    Keyword Args:
        fidelities (Dict[str, float]): Dictionary of gate fidelities. Each key
            is a string which specifies the gate and each value is the
            fidelity of that gate. When this argument is provided, folded
            gates contribute an amount proportional to their infidelity
            (1 - fidelity) to the total noise scaling. Fidelity values must be
            in the interval (0, 1]. Gates not specified have a default
            fidelity of 0.99**n where n is the number of qubits the gates act
            on.

            Supported gate keys are listed in the following table.::

                Gate key    | Gate
                -------------------------
                "H"         | Hadamard
                "X"         | Pauli X
                "Y"         | Pauli Y
                "Z"         | Pauli Z
                "I"         | Identity
                "CNOT"      | CNOT
                "CZ"        | CZ gate
                "TOFFOLI"   | Toffoli gate
                "single"    | All single qubit gates
                "double"    | All two-qubit gates
                "triple"    | All three-qubit gates

            Keys for specific gates override values set by "single", "double",
            and "triple".

            For example, `fidelities = {"single": 1.0, "H", 0.99}` sets all
            single-qubit gates except Hadamard to have fidelity one.

        squash_moments (bool): If True, all gates (including folded gates) are
            placed as early as possible in the circuit. If False, new moments
            are created for folded gates. This option only applies to QPROGRAM
            types which have a "moment" or "time" structure. Default is True.

        return_mitiq (bool): If True, returns a mitiq circuit instead of
            the input circuit type (if different). Default is False.

    Returns:
        folded: The folded quantum circuit as a QPROGRAM.

    """

    weight_mask = _create_weight_mask(circuit, kwargs.get("fidelities"))

    num_folds_mask = _create_fold_mask(
        weight_mask, scale_factor, folding_method="from_right"
    )

    return _apply_fold_mask(
        circuit, num_folds_mask, squash_moments=kwargs.get("squash_moments"),
    )


@noise_scaling_converter
def fold_gates_at_random(
    circuit: QPROGRAM,
    scale_factor: float,
    seed: Optional[int] = None,
    **kwargs: Any,
) -> QPROGRAM:
    r"""Returns a new folded circuit by applying the map G -> G G^dag G to a
    subset of gates of the input circuit, starting with gates at the
    right (end) of the circuit.

    The folded circuit has a number of gates approximately equal to
    scale_factor * n where n is the number of gates in the input circuit.

    For equal gate fidelities, this function reproduces the local unitary
    folding method defined in equation (5) of [Giurgica_Tiron_2020_arXiv]_.

    Args:
        circuit: Circuit to fold.
        scale_factor: Factor to scale the circuit by. Any real number >= 1.

    Keyword Args:
        fidelities (Dict[str, float]): Dictionary of gate fidelities. Each key
            is a string which specifies the gate and each value is the
            fidelity of that gate. When this argument is provided, folded
            gates contribute an amount proportional to their infidelity
            (1 - fidelity) to the total noise scaling. Fidelity values must be
            in the interval (0, 1]. Gates not specified have a default
            fidelity of 0.99**n where n is the number of qubits the gates act
            on.

            Supported gate keys are listed in the following table.::

                Gate key    | Gate
                -------------------------
                "H"         | Hadamard
                "X"         | Pauli X
                "Y"         | Pauli Y
                "Z"         | Pauli Z
                "I"         | Identity
                "CNOT"      | CNOT
                "CZ"        | CZ gate
                "TOFFOLI"   | Toffoli gate
                "single"    | All single qubit gates
                "double"    | All two-qubit gates
                "triple"    | All three-qubit gates

            Keys for specific gates override values set by "single", "double",
            and "triple".

            For example, `fidelities = {"single": 1.0, "H", 0.99}` sets all
            single-qubit gates except Hadamard to have fidelity one.

        squash_moments (bool): If True, all gates (including folded gates) are
            placed as early as possible in the circuit. If False, new moments
            are created for folded gates. This option only applies to QPROGRAM
            types which have a "moment" or "time" structure. Default is True.

        return_mitiq (bool): If True, returns a mitiq circuit instead of
            the input circuit type (if different). Default is False.

    Returns:
        folded: The folded quantum circuit as a QPROGRAM.

    """

    weight_mask = _create_weight_mask(circuit, kwargs.get("fidelities"))

    num_folds_mask = _create_fold_mask(
        weight_mask, scale_factor, folding_method="at_random", seed=seed,
    )

    return _apply_fold_mask(
        circuit, num_folds_mask, squash_moments=kwargs.get("squash_moments"),
    )
