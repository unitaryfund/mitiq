# Copyright (C) Unitary Fund
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

"""Utility functions for Pauli Twirling."""
import cirq
import numpy as np
import numpy.typing as npt
from cirq.circuits import Circuit

from mitiq.utils import matrix_to_vector, vector_to_matrix

pauli_unitary_list = [
    cirq.unitary((cirq.I)),
    cirq.unitary((cirq.X)),
    cirq.unitary((cirq.Y)),
    cirq.unitary((cirq.Z)),
]


def _n_qubit_paulis(num_qubits: int) -> list[npt.NDArray[np.complex64]]:
    """Get a list of n-qubit Pauli unitaries."""
    if not num_qubits >= 1:
        raise ValueError("Invalid number of qubits provided.")

    # get the n-qubit paulis from the Pauli group
    # disregard the n-qubit paulis with complex phase
    n_qubit_paulis = pauli_unitary_list
    for i in range(num_qubits - 1):
        n_qubit_paulis = n_qubit_paulis
        empty_pauli_list = []
        for j in pauli_unitary_list:
            for k in n_qubit_paulis:
                new_pauli = np.kron(j, k)
                empty_pauli_list.append(new_pauli)
        n_qubit_paulis = empty_pauli_list
    return n_qubit_paulis


def _pauli_vectorized_list(num_qubits: int) -> list[npt.NDArray[np.complex64]]:
    """Define a function to create a list of vectorized matrices.

    If the density matrix of interest has more than n>1 qubits, the
    Pauli group is used to generate n-fold tensor products before
    vectorizing the unitaries.
    """
    n_qubit_paulis = _n_qubit_paulis(num_qubits)
    output_pauli_vec_list = []
    for i in n_qubit_paulis:
        # the matrix_to_vector function stacks rows in vec form
        # transpose is used here to instead stack the columns
        matrix_trans = np.transpose(i)
        output_pauli_vec_list.append(matrix_to_vector(matrix_trans))
    return output_pauli_vec_list


def ptm_matrix(circuit: Circuit, num_qubits: int) -> npt.NDArray[np.complex64]:
    """Find the Pauli Transfer Matrix (PTM) of a circuit."""
    superop = cirq.kraus_to_superoperator(cirq.kraus(circuit))
    vec_pauli = _pauli_vectorized_list(num_qubits)
    n_qubit_paulis = _n_qubit_paulis(num_qubits)
    ptm_matrix = np.zeros([4**num_qubits, 4**num_qubits], dtype=complex)

    for i in range(len(vec_pauli)):
        superop_on_pauli_vec = np.matmul(superop, vec_pauli[i])
        superop_on_pauli_matrix_transpose = vector_to_matrix(
            superop_on_pauli_vec
        )
        superop_on_pauli_matrix = np.transpose(
            superop_on_pauli_matrix_transpose
        )

        ptm_matrix_row = []
        for j in n_qubit_paulis:
            pauli_superop_pauli = np.matmul(j, superop_on_pauli_matrix)
            ptm_matrix_row.append(
                (0.5**num_qubits) * np.trace(pauli_superop_pauli)
            )
        ptm_matrix[i] = ptm_matrix_row

    return ptm_matrix
