from mitiq import QPROGRAM, pec
from mitiq.pec.types import OperationRepresentation, NoisyOperation
from mitiq.interface import convert_to_mitiq, convert_from_mitiq

from cirq import (
Operation,
X,
Y,
Z,
Circuit,
)

from scipy import minimize
import numpy as np

def learn_representations(operation, executor, observable, 
        num_training_circuits: int = 10, epsilon0=0, eta0=1):
    """ 
    Args:
        operation: Single operation from the quantum program for which the 
                    quasiprobability representation will be learned
        executor: Function that executes the quantum program
        observable: 
    Returns:
        The quasiprobability representation learned from Clifford training circuit 
        data
    """
    
    from mitiq.cdr import generate_training_circuits 

    # Generate the Clifford training data
    training_circuits = generate_training_circuits(
        circuit=operation,
        num_training_circuits=num_training_circuits,
        fraction_non_clifford=0,
        method_select="uniform",
        method_replace="closest",
    )

    ideal_values = [ ]
    for training_circuit in training_circuits:
        ideal_values.append(executor(training_circuit))
    
    circ, in_type = convert_to_mitiq(operation)
    qubits = circ.all_qubits()

    def calculate_quasiprob_representation(epsilon, eta, qubits):
        """Compute quasi-probability representation by inverting local noise channel
         Args:
        operation: Single operation from the quantum program for which the 
                    quasiprobability representation will be learned
        executor: Function that executes the quantum program
        observable: 
    Returns:
        The quasiprobability representation learned from Clifford training circuit 
        data
        epsilon is the local noise strength, one of two optimization parameters 
        eta is the noise bias between reduced dephasing and depolarizing channels, 
        the other of the two optimization parameters"""
        if len(qubits) == 1:
            q = tuple(qubits)[0]
        
            alpha_1 = 1 + 3 * epsilon * (eta + 1) / (3 * (1 - epsilon) * (eta + 1) + 
                    epsilon * (3 * eta + 1))
            alpha_2 =  - epsilon / (3 * (1 - epsilon) * (eta + 1) + 
                    epsilon * (3 * eta + 1))
            alpha_3 = - epsilon*(3 * eta + 1) / (3 * (1 - epsilon) * (eta + 1) + 
                    epsilon * (3 * eta + 1))

            alphas = [alpha_1] +  12 * [alpha_2] + 3 * [alpha_3]
            post_ops = [[]]  # for eta_1, we do nothing, rather than I
            post_ops += [[P(q)] for P in [X, Y, Z]]  # 1Q Paulis


        # The two-qubit case: linear combination of 2Q Paulis
        elif len(qubits) == 2:
            q0, q1 = qubits

            alpha_1 = 1 + 15 *epsilon*(eta+1) / (15 * (1 - epsilon) * (eta + 1) + 
                    epsilon * (5 * eta + 1))
            alpha_2 = - epsilon / (15 * (1 - epsilon) * (eta + 1) + 
                    epsilon * (5 * eta + 1))
            alpha_3 = - epsilon * (5 * eta + 1)/(15 * (1 - epsilon) * (eta + 1) + 
                    epsilon*(5 * eta + 1))

            alphas = [alpha_1] + 15 * [alpha_2] + 3 *[alpha_3]
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
        imp_op_circuits = [convert_from_mitiq(c, in_type) for c in imp_op_circuits]

        # Build basis expansion.
        expansion = {NoisyOperation(c): a for c, a in zip(imp_op_circuits, alphas)}
        return OperationRepresentation(operation, expansion)


    def loss_function(epsilon, eta, qubits, ideal_values):    
        mitigated_value = pec.execute_with_pec(
                circuit=operation, 
                observable=observable,                                            	
                executor=executor,
            	representations=calculate_quasiprob_representation(epsilon, eta, qubits))

        num_train = len(ideal_values)
        return sum((mitigated_value*np.ones - ideal_values)**2)/num_train
    
    [epsilon_opt, eta_opt] = minimize(loss_function, [epsilon0, eta0], 
                        args=(qubits, ideal_values), method="BFGS")
        
        
    return calculate_quasiprob_representation(epsilon_opt, eta_opt, qubits)

