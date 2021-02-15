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
#  Some of them could be moved in future new sub-modules of PEC
#  (e.g. decomposition, tomo, etc.)

"""Utilities related to probabilistic error cancellation."""

from copy import deepcopy

import numpy as np

from cirq import (
    Circuit,
    OP_TREE,
    H,
    CNOT,
    LineQubit,
    DensityMatrixSimulator,
)


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
        operation_tree: Nested list of operations.
    Returns:
        The density matrix of the Choi state associated to the input circuit.
    """
    circuit = Circuit(operation_tree)
    return _circuit_to_choi(circuit)
