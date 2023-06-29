# Copyright (C) Unitary Fund
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the Quantum Subspace Expansion top level API."""

from typing import List

import numpy as np

import cirq

from mitiq import PauliString, Observable, QPROGRAM

from mitiq.qse import get_projector
from mitiq.qse.qse_utils import (
    _compute_overlap_matrix,
    _compute_hamiltonian_overlap_matrix,
)

from mitiq.interface import convert_to_mitiq

from mitiq.interface.mitiq_cirq import compute_density_matrix


def execute_with_depolarized_noise(circuit: QPROGRAM) -> np.ndarray:
    return compute_density_matrix(
        convert_to_mitiq(circuit)[0],
        noise_model=cirq.depolarize,
        noise_level=(0.01,),
    )


def execute_no_noise(circuit: QPROGRAM) -> np.ndarray:
    return compute_density_matrix(
        convert_to_mitiq(circuit)[0], noise_level=(0,)
    )


def test_get_projector():
    qc, _ = prepare_logical_0_state_for_5_1_3_code()
    (
        check_operators,
        code_hamiltonian,
    ) = get_5_1_3_code_check_operators_and_code_hamiltonian()
    P = get_projector(qc, execute_no_noise, check_operators, code_hamiltonian)
    uniform_projector = Observable(*[-0.25 * c for c in check_operators])
    assert P == uniform_projector


def test_compute_overlap_matrix():
    qc, _ = prepare_logical_0_state_for_5_1_3_code()
    (
        check_operators,
        _,
    ) = get_5_1_3_code_check_operators_and_code_hamiltonian()
    S = _compute_overlap_matrix(qc, execute_no_noise, check_operators, {})
    assert np.allclose(S, np.ones(16))

    S = _compute_overlap_matrix(
        qc, execute_with_depolarized_noise, check_operators, {}
    )
    # assert that S's diagonal is all ones but the off-diagonal elements
    #  are less than 1.
    # Diagonal terms are all 1's because we are computing
    # <Ψ|C_i C_i|Ψ> = <Ψ|Ψ> = 1
    assert np.allclose(np.diag(np.diag(S)), np.eye(16))
    # Check that all off diagonal entries of S are the same and less than 1
    # Off diagonal terms are less than 1 because we are computing <Ψ|C_i C_j|Ψ>
    # = <Ψ|C_k|Ψ> < 1 (since |Ψ> was rotated a bit from the logical subspace)
    # All off diagonal terms are the same because of the symmetry of the
    # total depolarizing noise.
    off_diag_elements = S[np.where(~np.eye(16, dtype=bool))]
    np.allclose(off_diag_elements, off_diag_elements[0])
    assert off_diag_elements[0] < 1


def test_compute_hamiltonian_overlap_matrix():
    qc, _ = prepare_logical_0_state_for_5_1_3_code()
    (
        check_operators,
        code_hamiltonian,
    ) = get_5_1_3_code_check_operators_and_code_hamiltonian()

    # If we have a full set of check operators that form a group then all
    # entries of the H matrix should be the same.
    # H_jk = sum over all i's <Ψ|C_i|Ψ>

    H = _compute_hamiltonian_overlap_matrix(
        qc, execute_no_noise, check_operators, code_hamiltonian, {}
    )
    assert np.allclose(H, -16 * np.ones(16))

    H = _compute_hamiltonian_overlap_matrix(
        qc,
        execute_with_depolarized_noise,
        check_operators,
        code_hamiltonian,
        {},
    )
    assert np.allclose(H, H[0][0])
    assert H[0][0].real > -16


def get_5_1_3_code_check_operators_and_code_hamiltonian() -> tuple:
    """
    Returns the check operators and code Hamiltonian for the [[5,1,3]] code
    The check operators are computed from the stabilizer generators:
    (1+G1)(1+G2)(1+G3)(1+G4)  G = [XZZXI, IXZZX, XIXZZ, ZXIXZ]
    source: https://en.wikipedia.org/wiki/Five-qubit_error_correcting_code
    """
    Ms = [
        "YIYXX",
        "ZIZYY",
        "IXZZX",
        "ZXIXZ",
        "YYZIZ",
        "XYIYX",
        "YZIZY",
        "ZZXIX",
        "XZZXI",
        "ZYYZI",
        "IYXXY",
        "IZYYZ",
        "YXXYI",
        "XXYIY",
        "XIXZZ",
        "IIIII",
    ]
    Ms_as_pauliStrings = [
        PauliString(M, coeff=1, support=range(5)) for M in Ms
    ]
    negative_Ms_as_pauliStrings = [
        PauliString(M, coeff=-1, support=range(5)) for M in Ms
    ]
    Hc = Observable(*negative_Ms_as_pauliStrings)
    return Ms_as_pauliStrings, Hc


def prepare_logical_0_state_for_5_1_3_code():
    """
    To simplify the testing logic. We hardcode the the logical 0 and logical 1
    states of the [[5,1,3]] code, copied from:
    https://en.wikipedia.org/wiki/Five-qubit_error_correcting_code
    We then use Gram-Schmidt orthogonalization to fill up the rest of the
    matrix with orthonormal vectors.
    Following this we construct a circuit that has this matrix as its gate.
    """

    def gram_schmidt(
        orthogonal_vecs: List[np.ndarray],
    ) -> np.ndarray:
        # normalize input
        orthonormalVecs = [
            vec / np.sqrt(np.vdot(vec, vec)) for vec in orthogonal_vecs
        ]
        dim = np.shape(orthogonal_vecs[0])[0]  # get dim of vector space
        for i in range(dim - len(orthogonal_vecs)):
            new_vec = np.zeros(dim)
            new_vec[i] = 1  # construct ith basis vector
            projs = sum(
                [
                    np.vdot(new_vec, cached_vec) * cached_vec
                    for cached_vec in orthonormalVecs
                ]
            )  # sum of projections of new vec with all existing vecs
            new_vec -= projs
            orthonormalVecs.append(
                new_vec / np.sqrt(np.vdot(new_vec, new_vec))
            )
        return np.reshape(orthonormalVecs, (32, 32)).T

    logical_0_state = np.zeros(32)
    for z in ["00000", "10010", "01001", "10100", "01010", "00101"]:
        logical_0_state[int(z, 2)] = 1 / 4
    for z in [
        "11011",
        "00110",
        "11000",
        "11101",
        "00011",
        "11110",
        "01111",
        "10001",
        "01100",
        "10111",
    ]:
        logical_0_state[int(z, 2)] = -1 / 4

    logical_1_state = np.zeros(32)
    for z in ["11111", "01101", "10110", "01011", "10101", "11010"]:
        logical_1_state[int(z, 2)] = 1 / 4
    for z in [
        "00100",
        "11001",
        "00111",
        "00010",
        "11100",
        "00001",
        "10000",
        "01110",
        "10011",
        "01000",
    ]:
        logical_1_state[int(z, 2)] = -1 / 4

    # Fill up the rest of the matrix with orthonormal vectors
    matrix = gram_schmidt([logical_0_state, logical_1_state])
    circuit = cirq.Circuit()
    g = cirq.MatrixGate(matrix)
    qubits = cirq.LineQubit.range(5)
    circuit.append(g(*qubits))
    return circuit, qubits
