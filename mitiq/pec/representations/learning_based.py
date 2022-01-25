from typing import (
    Optional,
    Callable,
    List,
)
import numpy as np
from scipy.optimize import minimize


from cirq import (
    X,
    Y,
    Z,
    Circuit,
)

from mitiq import pec, QPROGRAM, Observable
from mitiq._typing import QuantumResult
from mitiq.pec import OperationRepresentation, NoisyOperation
from mitiq.interface import convert_to_mitiq, convert_from_mitiq
from mitiq.cdr import generate_training_circuits


def learn_representations(operation: QPROGRAM,
                          ideal_executor: Callable[[QPROGRAM], QuantumResult],
                          noisy_executor: Callable[[QPROGRAM], QuantumResult],
                          observable: Optional[Observable] = None,
                          num_training_circuits: int = 10,
                          epsilon0: float = 0,
                          eta0: float = 1
                          ) -> OperationRepresentation:
    r""" Find optimal quasi-probability distribution of single operation
    with Clifford circuit training. Assume a combination of depolarizing
    and dephasing noise parametrized by the local noise strength
    :math:`\epsilon` and the noise bias :math:`\eta` between reduced dephasing
    and depolarizing channels.
    Args:
        operation: Single operation for which the
            quasiprobability representation will be learned.
        ideal_executor: Executes the training circuit and returns a
            `QuantumResult`.
        noisy_executor: Executes the single operation and returns a
            `QuantumResult`.
        observable: Observable to compute the expectation value of. If None,
            the `executor` must return an expectation value. Otherwise,
            the `QuantumResult` returned by `executor` is used to compute the
            expectation of the observable.
        num_training_circuits (optional): number of Clifford training circuits
            to be generated
        epsilon0 (optional): initial value for the local noise strength
        eta0 (optional): initial value for the noise bias
    Returns:
        The quasiprobability representation of the single (ideal) operation,
        learned from Clifford training circuit data
    """

    # Generate the Clifford training data
    circuit, in_type = convert_to_mitiq(operation)
    training_circuits = generate_training_circuits(
        circuit=circuit,
        num_training_circuits=num_training_circuits,
        fraction_non_clifford=0,
        method_select="uniform",
        method_replace="closest",
    )

    ideal_values = []
    for training_circuit in training_circuits:
        ideal_values.append(observable.expectation(training_circuit,
                                                   ideal_executor).real)

    def calculate_quasiprob_representations(circ: QPROGRAM, epsilon: float,
                                            eta: float
                                            ) -> OperationRepresentation:
        r"""Compute quasi-probability representation by inverting local noise channel
            and calculating basis expansion
        Args:
            circ: single operation, converted to Mitiq circuit
            epsilon: the local noise strength, an optimization parameter
            eta: the noise bias between reduced dephasing and depolarizing
                channels, an optimization parameter

        Returns:
            The quasiprobability representation of the single (ideal)
            operation, in terms of :math:`\epsilon` and the noise bias
            :math:`\eta`
        """
        qubits = circ.all_qubits()
        if len(qubits) == 1:
            q = tuple(qubits)[0]

            alpha_1 = 1 + 3 * epsilon * (eta + 1) / (3 * (1 - epsilon)
                                                     * (eta + 1) + epsilon
                                                     * (3 * eta + 1))
            alpha_2 = - epsilon / (3 * (1 - epsilon) * (eta + 1)
                                   + epsilon * (3 * eta + 1))
            alpha_3 = - epsilon * (3 * eta + 1) / (3 * (1 - epsilon)
                                                   * (eta + 1) + epsilon
                                                   * (3 * eta + 1))

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
            alpha_3 = - epsilon * (5 * eta + 1) / (15 * (1 - epsilon)
                                                   * (eta + 1)
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
        return OperationRepresentation(operation, expansion)

    def loss_function(params: List[np.ndarray], circ: QPROGRAM,
                      ideal_values: List[np.ndarray]) -> float:
        r""" Loss function: optimize the quasiprobability representation using
        the method of least squares
        Args:
        circ: single operation, converted to Mitiq circuit
        epsilon: the local noise strength, an optimization parameter
        eta: the noise bias between reduced dephasing and depolarizing
            channels, an optimization parameter
        ideal_values: expectation values obtained by simulations run on the
                    Clifford training circuits
        Returns: Square of the difference between the error-mitigated value and
        the ideal value, over the training set
        """
        epsilon = params[0]
        eta = params[1]

        representations = calculate_quasiprob_representations(circ, epsilon,
                                                              eta)
        mitigated_value = pec.execute_with_pec(
            circuit=circ,
            observable=observable,
            executor=noisy_executor,
            representations=[representations])

        num_train = len(ideal_values)
        return sum((mitigated_value * np.ones(num_train) - ideal_values) ** 2
                   ) / num_train

    x0 = [epsilon0, eta0]  # initial parameter values for optimization
    result = minimize(loss_function, x0, args=(circuit, ideal_values),
                      method="BFGS")
    x_result = result.x
    epsilon_opt = x_result[0]
    eta_opt = x_result[1]
    return calculate_quasiprob_representations(circuit, epsilon_opt, eta_opt)
