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

"""Functions for identity scaling on supported circuits."""
from copy import deepcopy
from typing import Any, List, Optional

import numpy as np
from cirq import GateOperation, Circuit, has_unitary
from cirq.ops import IdentityGate

from mitiq._typing import QPROGRAM
from mitiq.conversions import noise_scaling_converter
from mitiq.zne.scaling.folding import (
    _pop_measurements,
    _append_measurements,
    _create_weight_mask,
)


class UnscalableCircuitError(Exception):
    pass


def _check_scalable(circuit: Circuit) -> None:
    """Raises an error if the input circuit cannot be scaled by inserting
       identities.
    Args:
        circuit: Checks whether this circuit is able to be identity scaled.
    Raises:
        UnfoldableCircuitError:
            * If the circuit has intermediate measurements.
            * If the circuit has non-unitary channels which are not terminal
              measurements.
    """
    if not circuit.are_all_measurements_terminal():
        raise UnscalableCircuitError(
            "Circuit contains intermediate measurements and cannot be scaled"
            " by inserting identities."
        )

    if not has_unitary(circuit):
        raise UnscalableCircuitError(
            "Circuit contains non-unitary channels which are not terminal "
            "measurements and cannot be scaled by inserting identities."
        )


def _create_scale_mask(
    weight_mask: List[float],
    scale_factor: float,
    scaling_method: str,
    seed: Optional[int] = None,
) -> List[float]:
    r"""Returns a list of integers determining how many times an identity gate
    must be inserted to realize the desired input scale_factor.
    More precicely, the j_th element of the returned list is associated
    to the j_th gate G_j of a circuit that we want to scale and determines
    how many times I should be applied after G_j.
    The gate ordering is equal to the one used in the `all_operations()`
    method of the :class:`cirq.Circuit` class: gates from earlier moments
    come first and gates within the same moment follow the order in which
    they were given to the moment's constructor.
    The returned list is built such that the total weight of the
    folded circuit is approximately equal to scale_factor times the
    total weight of the input circuit.
    For equal weights, this function reproduces identity scaled version of
    the local unitary folding method defined in equation (5) of
    :cite:`Giurgica_Tiron_2020_arXiv`.
    Args:
        weight_mask: The weights of all the gates of the circuit to fold.
            Highly noisy gates should have a corresponding high weight.
            Gates with zero weight are assumed to be ideal and are not folded.
        scale_factor: The effective noise scale factor.
        scaling_method: A string equal to "at_random", or "from_left", or
            "from_right". Determines the partial folding method described in
            :cite:`Giurgica_Tiron_2020_arXiv`. If scale_factor is an odd
            integer, all methods are equivalent and this option is irrelevant.
        seed: A seed for the random number generator. This is used only when
            scaling_method is "at_random".
    Returns: The list of integers determining to how many times one should
            insert I in the circuit to be scaled.
    Example:
        >>>_create_scale_mask(
            weight_mask=[1.0, 0.5, 2.0, 0.0],
            scale_factor=4,
            scaling_method="from_left",
        )
        [2, 2, 1, 0]
    """

    if not 1.0 <= scale_factor:
        raise ValueError(
            f"Requires scale_factor >= 1.0 but scale_factor = {scale_factor}."
        )

    # Find the maximum integer smaller or equal to scale_factor
    num_uniform_inserts = int(scale_factor - 1.0)
    integer_scale_factor = num_uniform_inserts + 1

    # Uniformly scaling all gates to reach integer_scale_factor
    num_inserts_mask = []
    for w in weight_mask:
        if np.isclose(w, 0.0):
            num_inserts_mask.append(0)
        else:
            num_inserts_mask.append(num_uniform_inserts)

    # If the scale_factor is an integer, we are done.
    if np.isclose(integer_scale_factor, scale_factor, atol=0.01):
        return num_inserts_mask

    # Express scaling order through a list of indices
    scaling_order = list(range(len(weight_mask)))
    if scaling_method == "from_left":
        pass
    elif scaling_method == "from_right":
        scaling_order.reverse()
    elif scaling_method == "at_random":
        rnd_state = np.random.RandomState(seed)
        rnd_state.shuffle(scaling_order)
    else:
        raise ValueError(
            "The option 'scaling_method' is not valid."
            "It must be 'at_random', or 'from_left', or 'from_right'."
        )

    # Fold gates until the input scale_factor is better approximated
    input_circuit_weight = sum(weight_mask)
    output_circuit_weight = odd_integer_scale_factor * input_circuit_weight
    approx_error = np.abs(
        output_circuit_weight - scale_factor * input_circuit_weight
    )
    for j in scaling_order:
        # Skip gates with 0 weight
        if np.isclose(weight_mask[j], 0.0):
            continue
        # Compute the approx error if a new fold would be applied
        new_output_circuit_weight = output_circuit_weight + weight_mask[j]
        new_approx_error = np.abs(
            new_output_circuit_weight - scale_factor * input_circuit_weight
        )
        # Fold the candidate gate only if it helps improving the approximation
        if new_approx_error >= approx_error:
            break
        approx_error = new_approx_error
        output_circuit_weight = new_output_circuit_weight
        num_inserts_mask[j] += 1

    return num_inserts_mask


def _apply_scale_mask(
    circuit: Circuit,
    num_inserts_mask: List[int],
    squash_moments: Optional[bool] = True,
) -> Circuit:
    r"""Applies local identity scaling to the gates of the input circuit
    according to the input num_folds_mask.
    More precicely, I is applied after the j_th gate G_j of
    the input circuit an integer number of times given by num_folds_mask[j].
    The gate ordering is equal to the one used in the `all_operations()`
    method of the :class:`cirq.Circuit` class: gates from earlier moments
    come first and gates within the same moment follow the order in which
    they were given to the moment's constructor.
    Args:
        circuit: The quantum circuit to scale.
        num_inserts_mask: The list of integers indicating how many times
            the corresponding gates of 'circuit' should be folded.
        squash_moments: If True or None, all gates (including inserted gates)
            are placed as early as possible in the circuit. If False, new
            moments are created for inserted gates. This option only applies to
            QPROGRAM types which have a "moment" or "time" structure. Default
            is True.
    Returns: The scaled quantum circuit.
    """
    _check_scalable(circuit)
    circ_copy = deepcopy(circuit)
    measurements = _pop_measurements(circ_copy)

    num_gates = len(list(circ_copy.all_operations()))
    if num_gates != len(num_inserts_mask):
        raise ValueError(
            "The circuit and the scaling mask have incompatible sizes."
            f" The number of gates is {num_gates}"
            f" but len(num_folds_mask) is {len(num_inserts_mask)}."
        )

    scaled_circuit = circ_copy[:0]
    if squash_moments:
        for op, num_inserts in zip(
            circ_copy.all_operations(), num_inserts_mask
        ):
            scaled_circuit.append(
                [op]
                + num_inserts
                * [GateOperation(IdentityGate(len(op.qubits)), op.qubits)]
            )
    else:
        mask_index = 0
        for moment in circ_copy:
            scaled_moment = Circuit(moment)
            for op in moment:
                num_inserts = num_inserts_mask[mask_index]
                scaled_moment.append(
                    num_inserts
                    * [GateOperation(IdentityGate(len(op.qubits)), op.qubits)]
                )
                mask_index += 1
            # New scaled gates are only squashed with respect to scaled_moment
            # while scaled_circuit is not squashed.
            scaled_circuit.append(scaled_moment)

    _append_measurements(scaled_circuit, measurements)
    return scaled_circuit


@noise_scaling_converter
def scale_gates_from_left(
    circuit: QPROGRAM, scale_factor: float, **kwargs: Any
) -> QPROGRAM:
    """Returns a new scaled circuit by applying the map G -> G I to a
    subset of gates of the input circuit, starting with gates at the
    left (beginning) of the circuit.
    The scaled circuit has a number of gates approximately equal to
    scale_factor * n where n is the number of gates in the input circuit.
    For equal gate fidelities, this function reproduces the identoty scaled
    version of local unitary folding method defined in equation (5) of
    :cite:`Giurgica_Tiron_2020_arXiv`.
    Args:
        circuit: Circuit to scale.
        scale_factor: Factor to scale the circuit by. Any real number >= 1.
    Keyword Args:
        fidelities (Dict[str, float]): Dictionary of gate fidelities. Each key
            is a string which specifies the gate and each value is the
            fidelity of that gate. When this argument is provided, scaled
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
        squash_moments (bool): If True, all gates (including scaled gates) are
            placed as early as possible in the circuit. If False, new moments
            are created for scaled gates. This option only applies to QPROGRAM
            types which have a "moment" or "time" structure. Default is True.
        return_mitiq (bool): If True, returns a Mitiq circuit instead of
            the input circuit type (if different). Default is False.
    Returns:
        scaled: The scaled quantum circuit as a QPROGRAM.
    """

    weight_mask = _create_weight_mask(circuit, kwargs.get("fidelities"))

    num_inserts_mask = _create_scale_mask(
        weight_mask, scale_factor, scaling_method="from_left"
    )

    return _apply_scale_mask(
        circuit,
        num_inserts_mask,
        squash_moments=kwargs.get("squash_moments", True),
    )


@noise_scaling_converter
def scale_gates_from_right(
    circuit: QPROGRAM, scale_factor: float, **kwargs: Any
) -> QPROGRAM:
    r"""Returns a new scaled circuit by applying the map G -> G I to a
    subset of gates of the input circuit, starting with gates at the
    right (end) of the circuit.
    The scaled circuit has a number of gates approximately equal to
    scale_factor * n where n is the number of gates in the input circuit.
    For equal gate fidelities, this function reproduces the identity scaled
    version of local unitary folding method defined in equation (5) of
    :cite:`Giurgica_Tiron_2020_arXiv`.
    Args:
        circuit: Circuit to scale.
        scale_factor: Factor to scale the circuit by. Any real number >= 1.
    Keyword Args:
        fidelities (Dict[str, float]): Dictionary of gate fidelities. Each key
            is a string which specifies the gate and each value is the
            fidelity of that gate. When this argument is provided, scaled
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
        squash_moments (bool): If True, all gates (including scaled gates) are
            placed as early as possible in the circuit. If False, new moments
            are created for scaled gates. This option only applies to QPROGRAM
            types which have a "moment" or "time" structure. Default is True.
        return_mitiq (bool): If True, returns a Mitiq circuit instead of
            the input circuit type (if different). Default is False.
    Returns:
        scaled: The scaled quantum circuit as a QPROGRAM.
    """

    weight_mask = _create_weight_mask(circuit, kwargs.get("fidelities"))

    num_inserts_mask = _create_scale_mask(
        weight_mask, scale_factor, scaling_method="from_right"
    )

    return _apply_scale_mask(
        circuit,
        num_inserts_mask,
        squash_moments=kwargs.get("squash_moments", True),
    )


@noise_scaling_converter
def scale_gates_at_random(
    circuit: QPROGRAM,
    scale_factor: float,
    seed: Optional[int] = None,
    **kwargs: Any,
) -> QPROGRAM:
    r"""Returns a new scaled circuit by applying the map G -> G I to a
    subset of gates of the input circuit, starting with gates at the
    right (end) of the circuit.
    The scaled circuit has a number of gates approximately equal to
    scale_factor * n where n is the number of gates in the input circuit.
    For equal gate fidelities, this function reproduces the identity scaled
    version of local unitary folding method defined in equation (5) of
    :cite:`Giurgica_Tiron_2020_arXiv`.
    Args:
        circuit: Circuit to scale.
        scale_factor: Factor to scale the circuit by. Any real number >= 1.
    Keyword Args:
        fidelities (Dict[str, float]): Dictionary of gate fidelities. Each key
            is a string which specifies the gate and each value is the
            fidelity of that gate. When this argument is provided, scaled
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
        squash_moments (bool): If True, all gates (including scaled gates) are
            placed as early as possible in the circuit. If False, new moments
            are created for scaled gates. This option only applies to QPROGRAM
            types which have a "moment" or "time" structure. Default is True.
        return_mitiq (bool): If True, returns a Mitiq circuit instead of
            the input circuit type (if different). Default is False.
    Returns:
        scaled: The scaled quantum circuit as a QPROGRAM.
    """

    weight_mask = _create_weight_mask(circuit, kwargs.get("fidelities"))

    num_inserts_mask = _create_scale_mask(
        weight_mask, scale_factor, scaling_method="at_random", seed=seed,
    )

    return _apply_scale_mask(
        circuit,
        num_inserts_mask,
        squash_moments=kwargs.get("squash_moments", True),
    )
