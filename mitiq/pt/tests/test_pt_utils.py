# Copyright (C) Unitary Fund
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

import cirq
import numpy as np
import pytest

from mitiq.pt.pt_utils import (
    _n_qubit_paulis,
    _pauli_vectorized_list,
    ptm_matrix,
)

single_qubit_paulis = [
    cirq.unitary((cirq.I)),
    cirq.unitary((cirq.X)),
    cirq.unitary((cirq.Y)),
    cirq.unitary((cirq.Z)),
]

two_qubit_paulis = [
    np.kron(cirq.unitary((cirq.I)), cirq.unitary((cirq.I))),
    np.kron(cirq.unitary((cirq.I)), cirq.unitary((cirq.X))),
    np.kron(cirq.unitary((cirq.I)), cirq.unitary((cirq.Y))),
    np.kron(cirq.unitary((cirq.I)), cirq.unitary((cirq.Z))),
    np.kron(cirq.unitary((cirq.X)), cirq.unitary((cirq.I))),
    np.kron(cirq.unitary((cirq.X)), cirq.unitary((cirq.X))),
    np.kron(cirq.unitary((cirq.X)), cirq.unitary((cirq.Y))),
    np.kron(cirq.unitary((cirq.X)), cirq.unitary((cirq.Z))),
    np.kron(cirq.unitary((cirq.Y)), cirq.unitary((cirq.I))),
    np.kron(cirq.unitary((cirq.Y)), cirq.unitary((cirq.X))),
    np.kron(cirq.unitary((cirq.Y)), cirq.unitary((cirq.Y))),
    np.kron(cirq.unitary((cirq.Y)), cirq.unitary((cirq.Z))),
    np.kron(cirq.unitary((cirq.Z)), cirq.unitary((cirq.I))),
    np.kron(cirq.unitary((cirq.Z)), cirq.unitary((cirq.X))),
    np.kron(cirq.unitary((cirq.Z)), cirq.unitary((cirq.Y))),
    np.kron(cirq.unitary((cirq.Z)), cirq.unitary((cirq.Z))),
]


@pytest.mark.parametrize(
    "num_qubits, expected_shape, expected_output",
    [
        (1, (2, 2), single_qubit_paulis),
        (2, (4, 4), two_qubit_paulis),
    ],
)
def test_size_n_qubit_paulis(num_qubits, expected_shape, expected_output):
    """Check shape of n-qubits Paulis and size of the n-qubit Pauli list."""
    calculated_n_qubit_paulis = _n_qubit_paulis(num_qubits)

    assert len(calculated_n_qubit_paulis) == 4**num_qubits

    for i in range(4**num_qubits):
        item_n_qubit_paulis = calculated_n_qubit_paulis[i]
        assert item_n_qubit_paulis.shape == expected_shape
        assert (item_n_qubit_paulis == expected_output[i]).all()


@pytest.mark.parametrize("num_qubits", (0, -1))
def test_n_qubit_paulis_error(num_qubits):
    """Check error is raised when the number of qubits is invalid."""
    with pytest.raises(ValueError, match="Invalid number of qubits provided."):
        _n_qubit_paulis(num_qubits)


def test_pauli_vectorized_list():
    """Checks n-qubit Paulis are stacked by column."""
    calculated_output = _pauli_vectorized_list(1)
    expected_output = [
        np.array([1.0, 0.0, 0.0, 1.0]),
        np.array([0.0 + 0.0j, 1.0 + 0.0j, 1.0 + 0.0j, 0.0 + 0.0j]),
        np.array([0.0 + 0.0j, 0.0 + 1.0j, 0.0 - 1.0j, 0.0 + 0.0j]),
        np.array([1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, -1.0 + 0.0j]),
    ]
    for i in range(4):
        assert (expected_output[i] == calculated_output[i]).all()


@pytest.mark.parametrize(
    "input_gate, expected_ptm",
    [
        # Pauli X
        (
            cirq.X,
            np.array(
                [
                    [1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                    [0.0 + 0.0j, 1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                    [0.0 + 0.0j, 0.0 + 0.0j, -1.0 + 0.0j, 0.0 + 0.0j],
                    [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, -1.0 + 0.0j],
                ]
            ),
        ),
        # Pauli Y
        (
            cirq.Y,
            np.array(
                [
                    [1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                    [0.0 + 0.0j, -1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                    [0.0 + 0.0j, 0.0 + 0.0j, 1.0 + 0.0j, 0.0 + 0.0j],
                    [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, -1.0 + 0.0j],
                ]
            ),
        ),
        # Pauli Z
        (
            cirq.Z,
            np.array(
                [
                    [1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                    [0.0 + 0.0j, -1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                    [0.0 + 0.0j, 0.0 + 0.0j, -1.0 + 0.0j, 0.0 + 0.0j],
                    [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 1.0 + 0.0j],
                ]
            ),
        ),
        # Identity
        (cirq.I, np.identity(4)),
        # Note: For the rotation gates, the PTM is a 3 x 3 spatial rotation
        # matrix with the first row and column filled with 0s for the 4 x 4
        # PTM. The -ve sign on Sin terms are switched in PTM for X and Z
        # rotations compared to the conventional X and Z rotation matrices.
        # X Rotation
        (
            cirq.Rx(rads=0.167 * np.pi),
            np.array(
                [
                    [1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                    [0.0 + 0.0j, 1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                    [0.0 + 0.0j, 0.0 + 0.0j, 0.8660254 + 0.0j, 0.5 + 0.0j],
                    [0.0 + 0.0j, 0.0 + 0.0j, -0.5 + 0.0j, 0.8660254 + 0.0j],
                ]
            ),
        ),
        # Y Rotation
        # The PTM for Y rotation has a transposed 3 x 3 rotation matrix
        (
            cirq.Ry(rads=0.167 * np.pi),
            np.array(
                [
                    [1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                    [0.0 + 0.0j, 0.8660254 + 0.0j, 0.0 + 0.0j, 0.5 + 0.0j],
                    [0.0 + 0.0j, 0.0 + 0.0j, 1.0 + 0.0j, 0.0 + 0.0j],
                    [0.0 + 0.0j, -0.5 + 0.0j, 0.0 + 0.0j, 0.8660254 + 0.0j],
                ]
            ),
        ),
        # Z Rotation
        (
            cirq.Rz(rads=0.167 * np.pi),
            np.array(
                [
                    [1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                    [0.0 + 0.0j, 0.8660254 + 0.0j, 0.5 + 0.0j, 0.0 + 0.0j],
                    [0.0 + 0.0j, -0.5 + 0.0j, 0.8660254 + 0.0j, 0.0 + 0.0j],
                    [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 1.0 + 0.0j],
                ]
            ),
        ),
        # Hadamard
        (
            cirq.H,
            np.array(
                [
                    [1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                    [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 1.0 + 0.0j],
                    [0.0 + 0.0j, 0.0 + 0.0j, -1.0 + 0.0j, 0.0 + 0.0j],
                    [0.0 + 0.0j, 1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                ]
            ),
        ),
    ],
)
def test_ptm_single_ideal_gate(input_gate, expected_ptm):
    q0 = cirq.LineQubit(0)
    circuit = cirq.Circuit()
    circuit.append(input_gate(q0))
    calculated_ptm = ptm_matrix(circuit, 1)
    np.testing.assert_array_almost_equal(
        calculated_ptm, expected_ptm, decimal=2
    )


@pytest.mark.parametrize(
    "input_channel, expected_ptm",
    [  # Depolarizing channel
        # Note: The non-zero !=1 elements in the matrix are technically
        # supposed to be 0.8 (but calculated to be 0.73) due to 1-p as
        # provided in Example 1 on page 12 of
        # https://arxiv.org/abs/1509.02921
        (
            cirq.DepolarizingChannel(p=0.2),
            np.array(
                [
                    [1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                    [0.0 + 0.0j, 0.8 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                    [0.0 + 0.0j, 0.0 + 0.0j, 0.8 + 0.0j, 0.0 + 0.0j],
                    [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.8 + 0.0j],
                ]
            ),
        ),
        # Bit Flip channel
        (
            cirq.bit_flip(p=0.2),
            np.array(
                [
                    [
                        np.sqrt(0.2) + np.sqrt(0.8),
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                    ],
                    [
                        0.0 + 0.0j,
                        np.sqrt(0.2) + np.sqrt(0.8),
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                    ],
                    [
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        np.sqrt(0.2) - np.sqrt(0.8),
                        0.0 + 0.0j,
                    ],
                    [
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        np.sqrt(0.2) - np.sqrt(0.8),
                    ],
                ]
            ),
        ),
    ],
)
def test_ptm_single_qubit_error_channel(input_channel, expected_ptm):
    q0 = cirq.LineQubit(0)
    circuit = cirq.Circuit()
    circuit.append(input_channel(q0))
    calculated_ptm = ptm_matrix(circuit, 1)
    assert abs(calculated_ptm - expected_ptm).all() <= 0.7


@pytest.mark.parametrize(
    "input_gate, expected_ptm",
    [
        (
            cirq.CNOT,
            np.array(
                [
                    [
                        0.99999988 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                    ],
                    [
                        0.0 + 0.0j,
                        0.99999988 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                    ],
                    [
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.99999988 + 0.0j,
                        0.0 + 0.0j,
                    ],
                    [
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.99999988 + 0.0j,
                    ],
                    [
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.99999988 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                    ],
                    [
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.99999988 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                    ],
                    [
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.99999988 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                    ],
                    [
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        -0.99999988 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                    ],
                    [
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.99999988 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                    ],
                    [
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.99999988 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                    ],
                    [
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        -0.99999988 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                    ],
                    [
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.99999988 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                    ],
                    [
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.99999988 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                    ],
                    [
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.99999988 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                    ],
                    [
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.99999988 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                    ],
                    [
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.99999988 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                    ],
                ]
            ),
        ),
    ],
)
def test_ptm_CNOT_CZ(input_gate, expected_ptm):
    q0 = cirq.LineQubit(0)
    q1 = cirq.LineQubit(1)
    circuit = cirq.Circuit()
    circuit.append(input_gate(q0, q1))
    calculated_ptm = ptm_matrix(circuit, 2)
    assert abs(calculated_ptm - expected_ptm).all() <= 0.7
