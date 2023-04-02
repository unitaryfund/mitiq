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

"""Tests for the Clifford data regression top-level API."""
import pytest

from typing import List

import numpy as np

import cirq
from cirq import X, Z, I

from mitiq import PauliString, Observable, QPROGRAM

from mitiq import SUPPORTED_PROGRAM_TYPES

from mitiq.subspace_expansion import (
    execute_with_subspace_expansion,
)
from mitiq.subspace_expansion.utils import (
    convert_from_cirq_PauliSum_to_Mitiq_Observable,
)

from mitiq.interface import convert_from_mitiq, convert_to_mitiq

from mitiq.interface.mitiq_cirq import compute_density_matrix


# Allow execution with any QPROGRAM for testing.
def execute(circuit: QPROGRAM) -> np.ndarray:
    return compute_density_matrix(convert_to_mitiq(circuit)[0])


def batched_execute(circuits) -> List[np.ndarray]:
    return [execute(circuit) for circuit in circuits]


def simulate(circuit: QPROGRAM) -> np.ndarray:
    return compute_density_matrix(
        convert_to_mitiq(circuit)[0], noise_level=(0,)
    )


@pytest.mark.parametrize("circuit_type", SUPPORTED_PROGRAM_TYPES.keys())
def test_execute_with_subspace_expansion(circuit_type):
    qc, actual_qubits = prepare_logical_0_state_for_5_1_3_code()
    qc = convert_from_mitiq(qc, circuit_type)
    (
        check_operators,
        code_hamiltonian,
    ) = get_check_operators_and_code_hamiltonian()
    observable = convert_from_cirq_PauliSum_to_Mitiq_Observable(
        get_observable_cirq_in_code_space(actual_qubits, [Z, Z, Z, Z, Z])
    )
    mitigated_value = execute_with_subspace_expansion(
        qc, execute, check_operators, code_hamiltonian, observable
    )
    assert abs(mitigated_value.real - 1) < 0.001

    observable = convert_from_cirq_PauliSum_to_Mitiq_Observable(
        get_observable_cirq_in_code_space(actual_qubits, [X, X, X, X, X])
    )
    mitigated_value = execute_with_subspace_expansion(
        qc, execute, check_operators, code_hamiltonian, observable
    )
    assert abs(mitigated_value.real - 0.5) < 0.001


def get_observable_cirq_in_code_space(
    actual_qubits, observable: list[cirq.PauliString]
):
    FIVE_I = cirq.PauliString(
        [I(actual_qubits[i]) for i in range(len(actual_qubits))]
    )
    projector_onto_code_space = [
        [X, Z, Z, X, I],
        [I, X, Z, Z, X],
        [X, I, X, Z, Z],
        [Z, X, I, X, Z],
    ]

    observable_in_code_space = FIVE_I
    all_paulis = projector_onto_code_space + [observable]
    for g in all_paulis:
        f = cirq.PauliString([(g[i])(actual_qubits[i]) for i in range(len(g))])
        observable_in_code_space *= 0.5 * (FIVE_I + f)
    return observable_in_code_space


def get_check_operators_and_code_hamiltonian() -> tuple:
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
    def gram_schmidt(
        orthogonal_vecs: List[np.ndarray],
    ):
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
        return orthonormalVecs

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
    res = gram_schmidt([logical_0_state, logical_1_state])
    mat = np.reshape(res, (32, 32)).T
    circuit = cirq.Circuit()
    g = cirq.MatrixGate(mat)
    actual_qubits = cirq.LineQubit.range(5)
    circuit.append(g(*actual_qubits))
    return circuit, actual_qubits
