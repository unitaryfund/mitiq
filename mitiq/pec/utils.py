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

# TODO: Functions which don't fit in pec.py and sampling.py are placed here.
# Some of them could be moved in future new sub-modules of PEC
# (e.g. decomposition, tomo, etc.)

"""Utilities related to probabilistic error cancellation."""

import numpy as np
from typing import Tuple, List, Dict

from copy import deepcopy

from cirq import (
    Circuit,
    Operation,
    OP_TREE,
    I,
    X,
    Y,
    Z,
    H,
    CNOT,
    LineQubit,
    DensityMatrixSimulator,
)

# Type definition for a decomposition dictionary.
# Keys are ideal operations.
# Values describe the associated decompositions.
DecompositionDict = Dict[Operation, List[Tuple[float, List[Operation]]]]


def _simple_pauli_deco_dict(
    base_noise: float, simplify_paulis: bool = False
) -> DecompositionDict:
    """Returns a simple hard-coded decomposition
    dictionary to be used for testing and protoptyping.

    The decomposition is compatible with one-qubit or
    two-qubit circuits involving only Pauli and CNOT gates.

    The keys of the output dictionary are Pauli and CNOT operations.

    The decomposition assumes that Pauli and CNOT operations,
    followed by local depolarizing noise, are implementable.

    Args:
        base_noise: The depolarizing noise level.
        simplify_paulis: If True, products of Paulis are simplified to a
            single Pauli. If False, Pauli sequences are not simplified.

    Returns:
        decomposition_dict: The decomposition dictionary.

    """
    # Initialize two qubits
    qreg = LineQubit.range(2)

    # Single-qubit Pauli operations
    i0 = I.on(qreg[0])
    x0 = X.on(qreg[0])
    y0 = Y.on(qreg[0])
    z0 = Z.on(qreg[0])
    i1 = I.on(qreg[1])
    x1 = X.on(qreg[1])
    y1 = Y.on(qreg[1])
    z1 = Z.on(qreg[1])
    single_paulis = [x0, y0, z0, x1, y1, z1]

    # Single-qubit decomposition coefficients
    epsilon = base_noise * 4 / 3
    c_neg = -(1 / 4) * epsilon / (1 - epsilon)
    c_pos = 1 - 3 * c_neg
    assert np.isclose(c_pos + 3 * c_neg, 1.0)

    # Single-qubit decomposition dictionary
    decomposition_dict = {}
    if simplify_paulis:
        # Hard-coded simplified gates
        decomposition_dict = {
            x0: [(c_pos, [x0]), (c_neg, [i0]), (c_neg, [z0]), (c_neg, [y0])],
            y0: [(c_pos, [y0]), (c_neg, [z0]), (c_neg, [i0]), (c_neg, [x0])],
            z0: [(c_pos, [z0]), (c_neg, [y0]), (c_neg, [x0]), (c_neg, [i0])],
            x1: [(c_pos, [x1]), (c_neg, [i1]), (c_neg, [z1]), (c_neg, [y1])],
            y1: [(c_pos, [y1]), (c_neg, [z1]), (c_neg, [i1]), (c_neg, [x1])],
            z1: [(c_pos, [z1]), (c_neg, [y1]), (c_neg, [x1]), (c_neg, [i1])],
        }
    else:
        for local_paulis in [[x0, y0, z0], [x1, y1, z1]]:
            for key in local_paulis:
                key_deco_pos = [(c_pos, [key])]
                key_deco_neg = [(c_neg, [key, op]) for op in local_paulis]
                decomposition_dict[key] = (
                    key_deco_pos + key_deco_neg  # type: ignore
                )

    # Two-qubit Paulis
    xx = [x0, x1]
    xy = [x0, y1]
    xz = [x0, z1]
    yx = [y0, x1]
    yy = [y0, y1]
    yz = [y0, z1]
    zx = [z0, x1]
    zy = [z0, y1]
    zz = [z0, z1]
    double_paulis = [xx, xy, xz, yx, yy, yz, zx, zy, zz]

    # Two-qubit decomposition coefficients (assuming local noise)
    c_pos_pos = c_pos * c_pos
    c_pos_neg = c_neg * c_pos
    c_neg_neg = c_neg * c_neg
    assert np.isclose(c_pos_pos + 6 * c_pos_neg + 9 * c_neg_neg, 1.0)

    cnot = CNOT.on(qreg[0], qreg[1])
    cnot_decomposition = [(c_pos_pos, [cnot])]
    for p in single_paulis:
        cnot_decomposition.append((c_pos_neg, [cnot] + [p]))
    for pp in double_paulis:
        cnot_decomposition.append((c_neg_neg, [cnot] + pp))  # type: ignore

    decomposition_dict[cnot] = cnot_decomposition  # type: ignore

    return decomposition_dict  # type: ignore


def get_coefficients(
    ideal_operation: Operation, decomposition_dict: DecompositionDict
) -> List[float]:
    """Extracts, from the input decomposition dictionary, the decomposition
    coefficients associated to the input ideal_operation.

    Args:
        ideal_operation: The input ideal operation.
        decomposition_dict: The input decomposition dictionary.

    Returns:
        The decomposition coefficients of the input operation.
    """
    op_decomp = decomposition_dict[ideal_operation]

    return [coeff_and_seq[0] for coeff_and_seq in op_decomp]


def get_imp_sequences(
    ideal_operation: Operation, decomposition_dict: DecompositionDict
) -> List[List[Operation]]:
    """Extracts, from the input decomposition dictionary, the list of
    implementable sequences associated to the input ideal_operation.

    Args:
        ideal_operation: The input ideal operation.
        decomposition_dict: The input decomposition dictionary.

    Returns:
        The list of implementable sequences.
    """
    op_decomp = decomposition_dict[ideal_operation]

    return [coeff_and_seq[1] for coeff_and_seq in op_decomp]


def get_one_norm(
    ideal_operation: Operation, decomposition_dict: DecompositionDict
) -> float:
    """Extracts, from the input decomposition dictionary, the one-norm
    (i.e. the sum of absolute values) of the the decomposition coefficients
    associated to the input ideal_operation.

    Args:
        ideal_operation: The input ideal operation.
        decomposition_dict: The input decomposition dictionary.

    Returns:
        The one-norm of the decomposition coefficients.
    """
    coeffs = get_coefficients(ideal_operation, decomposition_dict)
    return np.linalg.norm(coeffs, ord=1)


def get_probabilities(
    ideal_operation: Operation, decomposition_dict: DecompositionDict
) -> List[float]:
    """Evaluates, from the input decomposition dictionary, the normalized
    probability distribution associated to the input ideal_operation.

    Sampling implementable sequences with this distribution (taking
    into account the corresponding "sign") approximates the exact
    decomposition of the input ideal_operation.

    Args:
        ideal_operation: The input ideal operation.
        decomposition_dict: The input decomposition dictionary.

    Returns:
        The probability distribution suitable for Monte Carlo sampling.
    """
    coeffs = get_coefficients(ideal_operation, decomposition_dict)
    return list(np.abs(coeffs) / np.linalg.norm(coeffs, ord=1))


def _max_ent_state_circuit(num_qubits: int) -> Circuit:
    r"""Generates a circuit which prepares the maximally entangled state
    |\omega\rangle = U |0\rangle  = \sum_i |i\rangle \otimes |i\rangle .

    Args:
        num_qubits: The number of qubits on which the circuit is applied.
            It must be an even number because of the structure of a
            maximally entangled state.

    Returns:
        The circuits which prepares the state |\omega\rangle.

    Raises:
        Value error: if num_qubits is not an even positive integer.
    """

    if not isinstance(num_qubits, int) or num_qubits % 2 or num_qubits == 0:
        raise ValueError(
            "The argument 'num_qubits' must be an even and positive integer."
        )

    alice_reg = LineQubit.range(num_qubits // 2)
    bob_reg = LineQubit.range(num_qubits // 2, num_qubits)

    return Circuit(
        # Prepare alice_register in a uniform superposition
        H.on_each(*alice_reg),
        # Correlate alice_register with bob_register
        [CNOT.on(alice_reg[i], bob_reg[i]) for i in range(num_qubits // 2)],
    )


def _circuit_to_choi(circuit: Circuit) -> np.ndarray:
    """Returns the density matrix of the Choi state associated to the
    input circuit.

    The density matrix completely characterizes the quantum channel induced by
    the input circuit (including the effect of noise if present).

    Args:
        circuit: The input circuit.
    Returns:
        The density matrix of the Choi state associated to the input circuit.
    """
    simulator = DensityMatrixSimulator()
    num_qubits = len(circuit.all_qubits())
    # Copy and remove all operations
    full_circ = deepcopy(circuit)[0:0]
    full_circ += _max_ent_state_circuit(2 * num_qubits)
    full_circ += circuit
    return simulator.simulate(full_circ).final_density_matrix  # type: ignore


def _operation_to_choi(operation_tree: OP_TREE) -> np.ndarray:
    """Returns the density matrix of the Choi state associated to the
    input operation tree (e.g. a single operation or a sequence of operations).

    The density matrix completely characterizes the quantum channel induced by
    the input operation tree (including the effect of noise if present).

    Args:
        circuit: The input circuit.
    Returns:
        The density matrix of the Choi state associated to the input circuit.
    """
    circuit = Circuit(operation_tree)
    return _circuit_to_choi(circuit)
