"""Functions for local and global unitary folding on supported circuits."""

from copy import deepcopy
from typing import Any, Callable, Iterable, List, Optional, Tuple, Union
import warnings

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


def squash_moments(circuit: Circuit) -> Circuit:
    """Returns a copy of the input circuit with all gates squashed into as few
    moments as possible.

    Args:
        circuit: Circuit to squash moments of.
    """
    return Circuit(
        circuit.all_operations(),
        strategy=InsertStrategy.EARLIEST,
        device=circuit.device
    )


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
        from mitiq.mitiq_qiskit.conversions import from_qiskit
        input_circuit_type = "qiskit"
        mitiq_circuit = from_qiskit(circuit)
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
        from mitiq.mitiq_qiskit.conversions import to_qiskit
        converted_circuit = to_qiskit(circuit)
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
        if kwargs.get("return_mitiq") is True:
            return fold_method(mitiq_circuit, *args, **kwargs)
        return convert_from_mitiq(
            fold_method(mitiq_circuit, *args, **kwargs), input_circuit_type
        )

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
    moment = circuit[moment_index]
    # Sometimes empty moments are generated when programs are converted into
    # cirq from other formats. These should not be folded.
    if len(moment) > 0:
        op = moment.operations[gate_index]
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


def _fold_gates(
    circuit: Circuit,
    moment_indices: Iterable[int],
    gate_indices: List[Iterable[int]],
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
        >>> _fold_gates(circuit, moment_indices=[0, 3],
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


def _fold_all_gates_locally(circuit: Circuit) -> None:
    """Replaces every gate G with G G^dag G by modifying the circuit in place.
    """
    _fold_moments(circuit, list(range(len(circuit))))


def _get_num_to_fold(scale_factor: float, ngates: int) -> int:
    """Returns the number of gates to fold to achieve the desired (approximate)
    scale factor.

    Args:
        scale_factor: Floating point value to scale the circuit by.
        ngates: Number of gates in the circuit to fold.
    """
    return int(round(ngates * (scale_factor - 1.0) / 2.0))


def _get_weight_for_gate(
        weights: Union[dict, None],
        op: ops.Operation
) -> float:
    """Returns the weight for a given gate, using a default value of 1.0 if
    weights is None or if the weight is not specified.

    Args:


    """
    if not weights:
        return 1.

    _cirq_gates_to_string_keys = {
        ops.H: "H",
        ops.X: "X",
        ops.Y: "Y",
        ops.Z: "Z",
        ops.T: "T",
        ops.I: "I",
        ops.CNOT: "CNOT",
        ops.CZ: "CZ",
        ops.TOFFOLI: "TOFFOLI"
    }

    weight = 1.

    if "single" in weights.keys() and len(op.qubits) == 1:
        weight = weights["single"]
    elif "double" in weights.keys() and len(op.qubits) == 2:
        weight = weights["double"]
    elif "triple" in weights.keys() and len(op.qubits) == 3:
        weight = weights["triple"]

    if op.gate in _cirq_gates_to_string_keys.keys():
        # Get the string key for this gate
        key = _cirq_gates_to_string_keys[op.gate]
        if key in weights.keys():
            weight = weights[_cirq_gates_to_string_keys[op.gate]]
    return weight


@converter
def fold_gates_from_left(
        circuit: QPROGRAM, scale_factor: float, **kwargs
) -> QPROGRAM:
    """Returns a new folded circuit by applying the map G -> G G^dag G to a
    subset of gates of the input circuit, starting with gates at the
    left (beginning) of the circuit.

    The folded circuit has a number of gates approximately equal to
    scale_factor * n where n is the number of gates in the input circuit.

    Args:
        circuit: Circuit to fold.
        scale_factor: Factor to scale the circuit by. Any real number >= 1.

    Keyword Args:
        weights (dict): Dictionary which defines the relative noise for each
            gate. Each key is a string which specifices the gate and each value
            is the weight for each gate. For example, the weight dictionary
            `weights = {"H": 1.0, "T": 0.5, "CNOT", 1.5}` means that T gates are
            50% as noisy as Hadamard gates, and CNOT gates are 3x as noisy as
            T gates. Gates not specified have a default weight of 1.0.

            Weights can be any positive floating point value. Supported gate
            keys are listed in the following table.

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

            Note: If specific gate keys are present, they override the weight
            for a "single", "double", or "triple". For example, if
            `weights = {"double": 1.5, "CNOT": 1.2}`, the weight of a CNOT is
            1.2 and the weight of every other two-qubit gates is 1.5.

        squash_moments (bool): If True, all gates (including folded gates) are
            placed as early as possible in the circuit. If False, new moments
            are created for folded gates. This option only applies to QPROGRAM
            types which have a "moment" or "time" structure. Default is True.

        return_mitiq (bool): If True, returns a mitiq circuit instead of
            the input circuit type (if different). Default is False.

    Returns:
        folded: The folded quantum circuit as a QPROGRAM.
    """
    if not circuit.are_all_measurements_terminal():
        raise ValueError(
            f"Input circuit contains intermediate measurements"
            " and cannot be folded."
        )

    if not 1. <= scale_factor:
        raise ValueError(
            f"Requires scale_factor >= 1 but scale_factor = {scale_factor}."
        )

    if scale_factor > 3.:
        return _fold_local(
            circuit, scale_factor, fold_method=fold_gates_from_left, **kwargs
        )

    weights = kwargs.get("weights")
    if weights and any(w < 0. for w in weights.values()):
        raise ValueError(
            "Negative weights were provided but weights should be non-negative."
        )

    folded = deepcopy(circuit)
    measurements = _pop_measurements(folded)

    ngates = len(list(folded.all_operations()))
    num_to_fold = _get_num_to_fold(scale_factor, ngates)
    if num_to_fold == 0:
        _append_measurements(folded, measurements)
        return folded

    weight_folded = 0.
    iteration = 0
    moment_shift = 0
    for (moment_index, moment) in enumerate(circuit):
        for gate_index in range(len(moment)):
            iteration += 1
            op = folded[moment_index + moment_shift].operations[gate_index]
            weight = _get_weight_for_gate(weights, op)
            if weight > 0.:
                _fold_gate_at_index_in_moment(
                    folded, moment_index + moment_shift, gate_index
                )
                moment_shift += 2
                weight_folded += weight

            if weight_folded >= num_to_fold or iteration == ngates:
                if weight_folded < num_to_fold:
                    warnings.warn(
                        "Scale factor not reached in folding",
                        category=RuntimeWarning
                    )
                _append_measurements(folded, measurements)
                if not (kwargs.get("squash_moments") is False):
                    folded = squash_moments(folded)
                return folded


@converter
def fold_gates_from_right(
        circuit: QPROGRAM, scale_factor: float, **kwargs
) -> Circuit:
    """Returns a new folded circuit by applying the map G -> G G^dag G
    to a subset of gates of the input circuit, starting with gates at
    the right (end) of the circuit.

    The folded circuit has a number of gates approximately equal to
    scale_factor * n where n is the number of gates in the input circuit.

    Args:
        circuit: Circuit to fold.
        scale_factor: Factor to scale the circuit by. Any real number >= 1.

    Keyword Args:
        weights (dict): Dictionary which defines the relative noise for each
            gate. Each key is a string which specifices the gate and each value
            is the weight for each gate. For example, the weight dictionary
            `weights = {"H": 1.0, "T": 0.5, "CNOT", 1.5}` means that T gates are
            50% as noisy as Hadamard gates, and CNOT gates are 3x as noisy as
            T gates. Gates not specified have a default weight of 1.0.

            Weights can be any positive floating point value. Supported gate
            keys are listed in the following table.

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

            Note: If specific gate keys are present, they override the weight
            for a "single", "double", or "triple". For example, if
            `weights = {"double": 1.5, "CNOT": 1.2}`, the weight of a CNOT is
            1.2 and the weight of every other two-qubit gates is 1.5.

        squash_moments (bool): If True, all gates (including folded gates) are
            placed as early as possible in the circuit. If False, new moments
            are created for folded gates. This option only applies to QPROGRAM
            types which have a "moment" or "time" structure. Default is True.

        return_mitiq (bool): If True, returns a mitiq circuit instead of
            the input circuit type (if different). Default is False.

    Returns:
        folded: The folded quantum circuit as a QPROGRAM.
    """
    if not circuit.are_all_measurements_terminal():
        raise ValueError(
            f"Input circuit contains intermediate measurements"
            " and cannot be folded."
        )
    circuit = deepcopy(circuit)
    measurements = _pop_measurements(circuit)

    reversed_circuit = Circuit(reversed(circuit))
    reversed_folded_circuit = fold_gates_from_left(
        reversed_circuit,
        scale_factor,
        weights=kwargs.get("weights"),
        squash_moments=False
    )
    folded = Circuit(reversed(reversed_folded_circuit))

    _append_measurements(folded, measurements)
    if not (kwargs.get("squash_moments") is False):
        folded = squash_moments(folded)
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
    """
    if moment_index_where_gate_was_folded not in moment_indices.keys():
        raise ValueError(
            f"Moment index {moment_index_where_gate_was_folded} not in moment"
            " indices."
        )
    for i in moment_indices.keys():
        moment_indices[i] += 2 * int(i >= moment_index_where_gate_was_folded)
    return moment_indices


@converter
def fold_gates_at_random(
    circuit: QPROGRAM, scale_factor: float, seed: Optional[int] = None, **kwargs
) -> QPROGRAM:
    """Returns a folded circuit by applying the map G -> G G^dag G to a random
    subset of gates in the input circuit.

    The folded circuit has a number of gates approximately equal to
    scale_factor * n where n is the number of gates in the input circuit.

    Args:
        circuit: Circuit to fold.
        scale_factor: Factor to scale the circuit by. Any real number >= 1.
        seed: [Optional] Integer seed for random number generator.


    Keyword Args:
        weights (dict): Dictionary which defines the relative noise for each
            gate. Each key is a string which specifices the gate and each value
            is the weight for each gate. For example, the weight dictionary
            `weights = {"H": 1.0, "T": 0.5, "CNOT", 1.5}` means that T gates are
            50% as noisy as Hadamard gates, and CNOT gates are 3x as noisy as
            T gates. Gates not specified have a default weight of 1.0.

            Weights can be any positive floating point value. Supported gate
            keys are listed in the following table.

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

            Note: If specific gate keys are present, they override the weight
            for a "single", "double", or "triple". For example, if
            `weights = {"double": 1.5, "CNOT": 1.2}`, the weight of a CNOT is
            1.2 and the weight of every other two-qubit gates is 1.5.

        squash_moments (bool): If True, all gates (including folded gates) are
            placed as early as possible in the circuit. If False, new moments
            are created for folded gates. This option only applies to QPROGRAM
            types which have a "moment" or "time" structure. Default is True.

        return_mitiq (bool): If True, returns a mitiq circuit instead of
            the input circuit type (if different). Default is False.

    Returns:
        folded: The folded quantum circuit as a QPROGRAM.
    """
    if not circuit.are_all_measurements_terminal():
        raise ValueError(
            f"Input circuit contains intermediate measurements"
            " and cannot be folded."
        )

    if not 1. <= scale_factor:
        raise ValueError(
            f"Requires scale_factor >= 1 but scale_factor = {scale_factor}."
        )

    if scale_factor > 3.:
        return _fold_local(
            circuit,
            scale_factor,
            fold_method=fold_gates_at_random,
            fold_method_args=(seed,),
            **kwargs
        )

    weights = kwargs.get("weights")
    if weights and any(w < 0. for w in weights.values()):
        raise ValueError(
            "Negative weights were provided but weights should be non-negative."
        )

    folded = deepcopy(circuit)
    measurements = _pop_measurements(folded)

    # Seed the random number generator
    rnd_state = np.random.RandomState(seed)

    ngates = len(list(folded.all_operations()))
    num_to_fold = _get_num_to_fold(scale_factor, ngates)
    if num_to_fold == 0:
        _append_measurements(folded, measurements)
        return folded

    # Keep track of where moments are in the folded circuit
    moment_indices = {i: i for i in range(len(folded))}

    # Keep track of which gates we can fold in each moment
    remaining_gate_indices = {
        moment: list(range(len(folded[moment])))
        for moment in range(len(folded)) if len(folded[moment]) > 0
    }

    # Any moment with at least one gate is fair game
    remaining_moment_indices = [
        i for i in remaining_gate_indices.keys() if remaining_gate_indices[i]
    ]

    weight_folded = 0.
    while remaining_moment_indices:
        # Get a moment index and gate index from the remaining set
        moment_index = rnd_state.choice(remaining_moment_indices)
        gate_index = rnd_state.choice(remaining_gate_indices[moment_index])

        # Get the weight of the gate at this moment index and gate index
        op = folded[moment_indices[moment_index]].operations[gate_index]
        weight = _get_weight_for_gate(weights, op)

        # Do the fold
        if weight > 0.:
            _fold_gate_at_index_in_moment(
                folded, moment_indices[moment_index], gate_index
            )
            weight_folded += weight

            # Update the moment indices for the folded circuit
            _update_moment_indices(moment_indices, moment_index)

        # Remove the current gate from the remaining set of gates to fold
        remaining_gate_indices[moment_index].remove(gate_index)

        # If there are no gates left in this moment, remove the moment
        if not remaining_gate_indices[moment_index]:
            remaining_moment_indices.remove(moment_index)

        if weight_folded >= num_to_fold:
            break

    if weight_folded < num_to_fold:
        warnings.warn(
            "Scale factor not reached in folding",
            category=RuntimeWarning
        )

    _append_measurements(folded, measurements)
    if not (kwargs.get("squash_moments") is False):
        folded = squash_moments(folded)
    return folded


def _fold_local(
    circuit: Circuit,
    scale_factor: float,
    fold_method: Callable[[Circuit, float, Tuple[Any]], Circuit],
    fold_method_args: Tuple[Any] = (),
    **kwargs
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
            f"The scale factor must be a real number greater than 1."
        )

    while scale_factor > 1.0:
        this_stretch = 3.0 if scale_factor > 3.0 else scale_factor
        folded = fold_method(
            folded, this_stretch, *fold_method_args, squash_moments=False
        )
        scale_factor /= 3.0

    if not (kwargs.get("squash_moments") is False):
        folded = squash_moments(folded)
    return folded


# Circuit level folding
@converter
def fold_global(circuit: QPROGRAM, scale_factor: float, **kwargs) -> QPROGRAM:
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
    if not (scale_factor >= 1):
        raise ValueError("The scale factor must be a real number >= 1.")

    if not circuit.are_all_measurements_terminal():
        raise ValueError(
            "Input circuit contains intermediate measurements"
            " and cannot be folded."
        )

    folded = deepcopy(circuit)
    measurements = _pop_measurements(folded)
    base_circuit = deepcopy(folded)

    # Determine the number of global folds and the final fractional scale
    num_global_folds, fraction_scale = divmod(scale_factor - 1, 2)
    # Do the global folds
    for _ in range(int(num_global_folds)):
        folded += Circuit(inverse(base_circuit), base_circuit)

    # Fold remaining gates until the scale is reached
    ops = list(base_circuit.all_operations())
    num_to_fold = int(round(fraction_scale * len(ops) / 2))

    if num_to_fold > 0:
        folded += Circuit([inverse(ops[-num_to_fold:])], [ops[-num_to_fold:]])

    _append_measurements(folded, measurements)
    if not (kwargs.get("squash_moments") is False):
        folded = squash_moments(folded)
    return folded
