from mitiq.cdr import generate_training_circuits

def learning_based_representations():
    from mitiq import QPROGRAM
    from mitiq.pec.types import OperationRepresentation, NoisyOperation
    from mitiq.interface import convert_to_mitiq, convert_from_mitiq

    from mitiq.pec.channels import tensor_product

    training_circuits = generate_training_circuits(
        circuit,
        num_training_circuits,
        fraction_non_clifford=0,
        method_select="uniform",
        method_replace="closest",
        random_state=random_state
    )


    return OperationRepresentation(ideal_operation, expansion)


def placeholder():
    ideal_values = [ ]
    mitigated_values = [ ]
    # randomly sample training circuits
    rnd_state = numpy.random.RandomState(fraction_selected)
    sampled_circuits = training_circuits[rnd_state]
    num_train = len(sampled_circuits)
    for n in num_train:
        ideal_values.append(ideal_executor(training_circuit))
    # Compute quasi-probability by inverting local noise channel
    eta_0 = ((1 - epsilon) * (eta + 1) + epsilon * (eta + 5))**-1
    eta_1 = 15 * epsilon * (eta + 1) * eta_0
    eta_2 = 5 * eta * eta_0 
    quasi_prob = (1 + eta_1) * np.identity(n) - epsilon * (eta_2 * mu_IZ + eta_0 *mu_IXYZ)
    mitigated_values.append(pec.execute_with_pec(circuit=training_circuit, 
            observable=observable, executor=noisy_executor, representations=quasi_prob))
    loss = sum((mitigated_values - ideal_values)**2)/num_train
    [epsilon_opt, eta_opt] = minimize(loss, [epsilon0, eta0], args=(), method="BFGS")
    return optimized_epsilon