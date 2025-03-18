from typing import List, cast, Optional
from mitiq.observable import Observable

import cirq
import numpy as np


def _copy_circuit_parallel(
    circuit: cirq.Circuit, num_copies: int = 2
) -> cirq.Circuit:
    """Copies a circuit num_copies times in parallel.

    Given a circuit that acts on N qubits,
    this function returns a circuit
    that copies the circuit num_copies times in parallel.
    This means the resulting circuit has N * num_copies qubits.

    Args:
        circuit:
            The circuit to be copied.
        num_copies:
            The number of copies of circuit to be made.

    Returns:
        A cirq circuit that is the parallel composition of
          num_copies copies of circuit.
    """

    new_circuit = cirq.Circuit()
    N = len(circuit.all_qubits())
    qubits = list(circuit.all_qubits())

    # LineQubits
    if isinstance(qubits[0], cirq.LineQubit):

        def map_for_line_qubits(q: cirq.Qid) -> cirq.Qid:
            assert isinstance(q, cirq.LineQubit)
            return cirq.LineQubit(q.x + N * i)

        for i in range(num_copies):
            new_circuit += circuit.transform_qubits(map_for_line_qubits)

    # GridQubits
    elif isinstance(qubits[0], cirq.GridQubit):
        qubits_cast_grid = cast(List[cirq.GridQubit], qubits)
        grid_rows = max([qu.row + 1 for qu in qubits_cast_grid])

        def map_for_grid_qubits(qu: cirq.Qid) -> cirq.Qid:
            assert isinstance(qu, cirq.GridQubit)
            return cirq.GridQubit(qu.row + grid_rows * i, qu.col)

        for i in range(num_copies):
            new_circuit += circuit.transform_qubits(map_for_grid_qubits)

    return new_circuit


def _apply_cyclic_system_permutation(
    matrix: np.array, N_qubits: int, M: int = 2
) -> np.array:
    """
    Function that shifts the rows of a matrix or vector in such a way,
    that each of the M registers of N_qubit qubits are shifted cyclically.
    The implementation is identical to left multiplication with repeated swap gates,
    however this optimisation in considerably faster.

    Args:
        matrix: The matrix or vector that should be shifted.
        N_qubits: The number of qubits in each register.
        M: The number of registers.

    Returns:
        The matrix or vector with the rows shifted cyclically.
    """
    matrix = np.array(matrix)

    # determine the row permutation for the cyclic shift operation
    permutation = [
        j + i
        for j in range(2**N_qubits)
        for i in range(0, 2 ** (M * N_qubits), 2**N_qubits)
    ]

    # Some fancy index magic to permute the rows in O(n) time and space (n=2**(M*N_qubits))
    idx = np.empty_like(permutation)
    idx[permutation] = np.arange(len(permutation))

    # allow to work with lists or arrays of np.ndarrays
    if matrix.ndim == 2:
        matrix = matrix[idx]
    elif matrix.ndim == 3:
        matrix[:] = matrix[:, idx]
    else:
        raise TypeError(
            "matrix must be a 2 dimensional array or a listor array of 2 dimensional arrays"
        )
    return matrix


def _apply_symmetric_observable(
    matrix: np.array,
    N_qubits: int,
    observable: Optional[Observable] = None,
    M: int = 2,
) -> np.array:
    """
    Function that applies a symmetric observable to a matrix or vector.
    The observable is assumed to be the Z observable by default.

    Args:
        matrix: The matrix or vector that should be shifted.
        N_qubits: The number of qubits in each register.
        observable: The observable that should be applied. If None, the Z observable is used.
        M: The number of registers.

    Returns:
        The matrix or vector with the observable applied.
    """
    if observable is None or np.allclose(
        observable, np.array([[1.0, 0.0], [0.0, -1.0]])
    ):
        # use the default Z observable
        sym_observable_diagonals = []
        for i in range(N_qubits):
            observable_i_diagonal = np.array(
                [
                    j
                    for k in range(2 ** (i))
                    for j in [1.0, -1.0]
                    for l in range(2 ** (N_qubits - i - 1))
                ]
            )

            # turn [a, b, c] into [a,a,a,b,b,b,c,c,c]. This is the same as tensoring the N_qubit identity on the right
            observable_i_diagonal_system1 = np.array(
                [observable_i_diagonal for _ in range(2**N_qubits)]
            ).flatten("F")
            # turn [a,b,c] into [a,b,c,a,b,c,a,b,c]. This is the same as tensoring the N_qubit identity on the left
            observable_i_diagonal_system2 = np.array(
                [observable_i_diagonal for _ in range(2**N_qubits)]
            ).flatten("C")
            # add the symmetric observable
            sym_observable_diagonals.append(
                0.5
                * (
                    observable_i_diagonal_system1
                    + observable_i_diagonal_system2
                )
            )

        if matrix.ndim == 2:
            return np.array([sod * matrix for sod in sym_observable_diagonals])
        elif matrix.ndim == 3:
            return np.array(
                [
                    sod * mat
                    for sod in sym_observable_diagonals
                    for mat in matrix
                ]
            )
        else:
            raise ValueError("matrix should be a 2D or 3D ndarray")

    else:
        sym_observable_matrices = []
        for i in range(N_qubits):
            observable_i_matrix = np.kron(
                np.kron(np.eye(2**i), observable),
                np.eye(2 ** (N_qubits - i - 1)),
            )
            sym_observable_matrix = 0.5 * (
                np.kron(observable_i_matrix, np.eye(2**N_qubits))
                + np.kron(np.eye(2**N_qubits), observable_i_matrix)
            )
            sym_observable_matrices.append(sym_observable_matrix)

    return np.array(sym_observable_matrices) @ matrix
