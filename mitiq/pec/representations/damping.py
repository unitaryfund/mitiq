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
"""Functions related to representations with amplitude damping noise."""

from typing import List
from itertools import product

import numpy as np
import numpy.typing as npt

from cirq import (
    Circuit,
    Z,
    kraus,
    AmplitudeDampingChannel,
    reset,
)

from mitiq.pec.types import OperationRepresentation, NoisyOperation

from mitiq.pec.channels import tensor_product


# TODO: this may be extended to an arbitrary QPROGRAM (GitHub issue gh-702).
def _represent_operation_with_amplitude_damping_noise(
    ideal_operation: Circuit,
    noise_level: float,
) -> OperationRepresentation:
    r"""Returns the quasi-probability representation of the input
    single-qubit ``ideal_operation`` with respect to a basis of noisy
    operations.

    Any ideal single-qubit unitary followed by local amplitude-damping noise
    of equal ``noise_level`` is assumed to be in the basis of implementable
    operations.

    The representation is based on the analytical result presented
    in :cite:`Takagi2020`.

    Args:
        ideal_operation: The ideal operation (as a QPROGRAM) to represent.
        noise_level: The noise level of each amplitude damping channel.

    Returns:
        The quasi-probability representation of the ``ideal_operation``.

    .. note::
        The input ``ideal_operation`` is typically a QPROGRAM with a single
        gate but could also correspond to a sequence of more gates.
        This is possible as long as the unitary associated to the input
        QPROGRAM, followed by a single final amplitude damping channel, is
        physically implementable.

    .. note::
        The input ``ideal_operation`` must be a ``cirq.Circuit``.
    """

    if not isinstance(ideal_operation, Circuit):
        raise NotImplementedError(
            "The input ideal_operation must be a cirq.Circuit.",
        )

    qubits = ideal_operation.all_qubits()

    if len(qubits) == 1:
        q = tuple(qubits)[0]

        eta_0 = (1 + np.sqrt(1 - noise_level)) / (2 * (1 - noise_level))
        eta_1 = (1 - np.sqrt(1 - noise_level)) / (2 * (1 - noise_level))
        eta_2 = -noise_level / (1 - noise_level)
        etas = [eta_0, eta_1, eta_2]
        post_ops = [[], Z(q), reset(q)]

    else:
        raise ValueError(  # pragma: no cover
            "Only single-qubit operations are supported."  # pragma: no cover
        )  # pragma: no cover

    # Basis of implementable operations as circuits
    imp_op_circuits = [ideal_operation + Circuit(op) for op in post_ops]

    # Build basis_expantion
    expansion = {NoisyOperation(c): a for c, a in zip(imp_op_circuits, etas)}

    return OperationRepresentation(ideal_operation, expansion)


def amplitude_damping_kraus(
    noise_level: float,
    num_qubits: int,
) -> List[npt.NDArray[np.complex64]]:
    """Returns the Kraus operators of the tensor product of local
    depolarizing channels acting on each qubit.
    """
    local_noisy_op = AmplitudeDampingChannel(noise_level)
    local_kraus = list(kraus(local_noisy_op))
    return [
        tensor_product(*kraus_string)
        for kraus_string in product(local_kraus, repeat=num_qubits)
    ]
