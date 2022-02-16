from typing import List

from cirq import (
    Operation,
    X,
    Y,
    Z,
    Circuit,
)

from mitiq import QPROGRAM
from mitiq.pec import OperationRepresentation, NoisyOperation
from mitiq.interface import convert_to_mitiq, convert_from_mitiq


def represent_operation_with_biased_noise(
    ideal_operation: QPROGRAM, epsilon: float, eta: float,
) -> OperationRepresentation:
    circ, in_type = convert_to_mitiq(ideal_operation)

    post_ops: List[List[Operation]]
    qubits = circ.all_qubits()

    if len(qubits) == 1:
        q = tuple(qubits)[0]

        alpha_1 = 1 + 3 * epsilon * (eta + 1) / (3 * (1 - epsilon)
                                                 * (eta + 1) + epsilon
                                                 * (3 * eta + 1))
        alpha_2 = - epsilon / (3 * (1 - epsilon) * (eta + 1)
                               + epsilon * (3 * eta + 1))
        alpha_3 = - epsilon * (3 * eta + 1) / (3 * (1 - epsilon) * (eta + 1)
                                               + epsilon * (3 * eta + 1))

        alphas = [alpha_1] + 12 * [alpha_2] + 3 * [alpha_3]
        post_ops = [[]]  # for eta_1, we do nothing, rather than I
        post_ops += [[P(q)] for P in [X, Y, Z]]  # 1Q Paulis

        # The two-qubit case: linear combination of 2Q Paulis
    elif len(qubits) == 2:
        q0, q1 = qubits

        alpha_1 = 1 + 15 * epsilon * (eta + 1) / (15 * (1 - epsilon)
                                                  * (eta + 1) + epsilon
                                                  * (5 * eta + 1))
        alpha_2 = - epsilon / (15 * (1 - epsilon) * (eta + 1)
                               + epsilon * (5 * eta + 1))
        alpha_3 = - epsilon * (5 * eta + 1) / (15 * (1 - epsilon) * (eta + 1)
                                               + epsilon * (5 * eta + 1))

        alphas = [alpha_1] + 15 * [alpha_2] + 3 * [alpha_3]
        post_ops = [[]]  # for eta_1, we do nothing, rather than I x I
        post_ops += [[P(q0)] for P in [X, Y, Z]]  # 1Q Paulis for q0
        post_ops += [[P(q1)] for P in [X, Y, Z]]  # 1Q Paulis for q1
        post_ops += [
            [Pi(q0), Pj(q1)] for Pi in [X, Y, Z] for Pj in [X, Y, Z]
        ]  # 2Q Paulis

    else:
        raise ValueError(
            "Can only represent single- and two-qubit gates."
            "Consider pre-compiling your circuit."
        )
    # Basis of implementable operations as circuits.
    imp_op_circuits = [circ + Circuit(op) for op in post_ops]

    # Convert back to input type.
    imp_op_circuits = [convert_from_mitiq(c, in_type)
                       for c in imp_op_circuits]

    # Build basis expansion.
    expansion = {NoisyOperation(c): a for c, a in zip(imp_op_circuits,
                                                      alphas)}
    return OperationRepresentation(ideal_operation, expansion)
