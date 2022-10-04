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

"""Functions for finding optimal representations given a noisy basis."""

from typing import cast, List, Optional

import numpy as np
import numpy.typing as npt
from scipy.optimize import minimize, LinearConstraint

from cirq import kraus

from mitiq import QPROGRAM
from mitiq.interface import convert_to_mitiq
from mitiq.pec.types import NoisyBasis, OperationRepresentation
from mitiq.pec.channels import matrix_to_vector, kraus_to_super


def minimize_one_norm(
    ideal_matrix: npt.NDArray[np.complex64],
    basis_matrices: List[npt.NDArray[np.complex64]],
    tol: float = 1.0e-8,
    initial_guess: Optional[npt.NDArray[np.float64]] = None,
) -> npt.NDArray[np.float64]:
    r"""
    Returns the list of real coefficients :math:`[x_0, x_1, \dots]`,
    which minimizes :math:`\sum_j |x_j|` with the contraint that
    the following representation of the input ``ideal_matrix`` holds:

    .. math::
        :nowrap:

        \text{ideal_matrix} = x_0 A_0 + x_1 A_1 + ...,

    where :math:`\{A_j\}` are the basis matrices, i.e., the elements of
    the input ``basis_matrices``.

    This function can be used to compute the optimal representation
    of an ideal superoperator (or Choi state) as a linear
    combination of real noisy superoperators (or Choi states).

    Args:
        ideal_matrix: The ideal matrix to represent.
        basis_matrices: The list of basis matrices.
        tol: The error tolerance for each matrix element
            of the represented matrix.
        initial_guess: Optional initial guess for the coefficients
            :math:`[x_0, x_1, \dots]`.

    Returns:
        The list of optimal coefficients :math:`[x_0, x_1, \dots]`.
    """

    # Map complex matrices to extended real matrices
    ideal_matrix_real = np.hstack(
        (np.real(ideal_matrix), np.imag(ideal_matrix))
    )
    basis_matrices_real = [
        np.hstack((np.real(mat), np.imag(mat))) for mat in basis_matrices
    ]

    # Express the representation constraint written in the docstring in the
    # form of a matrix multiplication applied to the x vector: A @ x == b.
    matrix_a = np.array(
        [
            matrix_to_vector(mat)  # type: ignore[arg-type]
            for mat in basis_matrices_real
        ]
    ).T
    array_b = matrix_to_vector(ideal_matrix_real)  # type: ignore[arg-type]

    constraint = LinearConstraint(matrix_a, lb=array_b - tol, ub=array_b + tol)

    def one_norm(x: npt.NDArray[np.complex64]) -> float:
        return cast(float, np.linalg.norm(x, 1))

    if initial_guess is None:
        initial_guess = np.zeros(len(basis_matrices))

    result = minimize(one_norm, x0=initial_guess, constraints=constraint)

    if not result.success:
        raise RuntimeError("The search for an optimal representation failed.")

    return result.x


def find_optimal_representation(
    ideal_operation: QPROGRAM,
    noisy_basis: NoisyBasis,
    tol: float = 1.0e-8,
    initial_guess: Optional[npt.NDArray[np.float64]] = None,
) -> OperationRepresentation:
    r"""Returns the ``OperationRepresentaiton`` of the input ideal operation
    which minimizes the one-norm of the associated quasi-probability
    distribution.

    More precicely, it solve the following optimization problem:

    .. math::
        \min_{{\eta_\alpha}} = \sum_\alpha |\eta_\alpha|,
        \text{ such that }
        \mathcal G = \sum_\alpha \eta_\alpha \mathcal O_\alpha,

    where :math:`\{\mathcal O_j\}` is the input basis of noisy operations.

    Args:
        ideal_operation: The ideal operation to represent.
        noisy_basis: The ``NoisyBasis`` in which the ``ideal_operation``
            should be represented. It must contain ``NoisyOperation`` objects
            which are initialized with a numerical superoperator matrix.
        tol: The error tolerance for each matrix element
            of the represented operation.
        initial_guess: Optional initial guess for the coefficients
            :math:`\{ \eta_\alpha \}``.

    Returns: The optimal OperationRepresentation.
    """
    ideal_cirq_circuit, _ = convert_to_mitiq(ideal_operation)
    ideal_matrix = kraus_to_super(
        cast(List[npt.NDArray[np.complex64]], kraus(ideal_cirq_circuit))
    )
    basis_set = noisy_basis.elements

    try:
        basis_matrices = [noisy_op.channel_matrix for noisy_op in basis_set]
    except ValueError as err:
        if str(err) == "The channel matrix is unknown.":
            raise ValueError(
                "The input noisy_basis should contain NoisyOperation objects"
                " which are initialized with a numerical superoperator matrix."
            )
        else:
            raise err  # pragma no cover

    # Run numerical optimization problem
    quasi_prob_dist = minimize_one_norm(
        ideal_matrix,
        basis_matrices,
        tol=tol,
        initial_guess=initial_guess,
    )

    basis_expansion = {op: eta for op, eta in zip(basis_set, quasi_prob_dist)}
    return OperationRepresentation(ideal_operation, basis_expansion)
