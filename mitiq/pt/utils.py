# Copyright (C) Unitary Fund
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

"""Utility functions for Pauli Twirling of CZ & CNOT gates."""

import cirq
import numpy as np


def _matrix_vec(matrix):
    """Define the vectorized form of an array.

    Transpose of a valid density matrix is converted into a
    column vector such that some superoperator can act on the
    vectorized form.
    """
    matrix_transpose = np.transpose(matrix)
    return np.reshape(matrix_transpose, (-1, 1))


def _matrix_no_vec(vec_matrix, num_qubits):
    """Define the matrix form of a vectorized array.

    Converts a column vector (vec) to a square matrix.
    """
    vec_to_matrix = np.reshape(vec_matrix, (2**num_qubits, 2**num_qubits))
    return np.transpose(vec_to_matrix)


def _pauli_vectorized_list(num_qubits):
    """Define a function to create a list of vectorized matrices.

    If the density matrix of interest has more than n>1 qubits, the
    Pauli group is used to generate n-fold tensor products before
    vectorizing the unitaries.
    """
    pauli_unitary_list = [
        cirq.unitary((cirq.I)),
        cirq.unitary((cirq.X)),
        cirq.unitary((cirq.Y)),
        cirq.unitary((cirq.Z)),
    ]

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

    output_pauli_vec_list = []
    for i in n_qubit_paulis:
        output_pauli_vec_list.append(_matrix_vec(i))
    return output_pauli_vec_list
