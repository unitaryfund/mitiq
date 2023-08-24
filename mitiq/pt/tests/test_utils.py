# Copyright (C) Unitary Fund
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

"""Tests for Pauli Twirling Utility functions."""

import random

import numpy as np

from mitiq.pt.utils import _matrix_no_vec, _matrix_vec, _pauli_vectorized_list


def test_matrix_to_vec():
    """Check a square matrix is vectorized into a column vector."""
    for i in range(1, 5):
        func_output = _matrix_vec(np.random.rand(2**i, 2**i))
        assert func_output.shape[-1] == 1


def test_matrix_no_vec():
    """Check a vec (column vector) is turned back into a square matrix."""
    for i in range(1, 5):
        vec_output = _matrix_vec(np.random.rand(2**i, 2**i))
        vec_to_matrix = _matrix_no_vec(vec_output, i)
        assert vec_to_matrix.shape[0] == vec_to_matrix.shape[-1]


def test_size_of_pauli_vec():
    """Check the output size of a list of n-qubit vec Paulis.

    Number of n-qubit Paulis should be  4**num_qubits.
    Size of each matrix should be 2**num_qubits.
    """
    for i in range(1, 5):
        expected_output_pauli_vec_list = _pauli_vectorized_list(i)
        assert len(expected_output_pauli_vec_list) == 4**i
        # randomly check 1 entry in the output is a column vector
        j = random.randrange(len(expected_output_pauli_vec_list) + 1)
        assert expected_output_pauli_vec_list[j].shape[-1] == 1
