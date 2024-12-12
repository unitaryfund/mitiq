
import random
import json
import jsonschema
from jsonschema import validate
from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error
from qiskit_aer import AerSimulator
from qiskit import transpile


def load(schema_path):
    with open(schema_path, 'r') as file:
        return json.load(file)
    
def add_random_pauli_identities(circuit, k):
    """Adds random Pauli gate pairs that act as identity on each qubit in the circuit. This is bc zne was throwing an error when that the cirucit was very short for zne to work.

    Parameters:
        circuit: qiskit circuit to modify.
        k (int): number of Pauli identity operations to add (must be even)
    
    Returns:
        qiskit circuit with added identities
    """

    circuit = circuit.copy()
    if k % 2 != 0:
        raise ValueError("k must be an even integer for the sequence to be equivalent to identity.")
    
    num_qubits = circuit.num_qubits
    pauli_pairs = [('x', 'x'), ('y', 'y'), ('z', 'z')]  # Possible Pauli identity pairs

    for qubit in range(num_qubits):
        for _ in range(k // 2):
            pauli_1, pauli_2 = random.choice(pauli_pairs)
            getattr(circuit, pauli_1)(qubit)
            getattr(circuit, pauli_2)(qubit)
    
    return circuit


def validate_experiment(experiment, schema):
    try:
        validate(instance=experiment, schema=schema)
        print("validation passed")
    except jsonschema.exceptions.ValidationError as e:
        print("validation failed")
    return None



def validate_composed_experiment(composed_experiment, schema):
    """This function validates a composed experiment against multiple schemas, ensuring the techniques used are compatible, accounting for their order as well

    Parameters:
        experiment: dicitionary containing the key "techniques" whose value is an array of experiments the user wishes to compose.
        schema: this is the schema to validate the individa

    """
    try:
        validate(instance=composed_experiment, schema=schema)
        print("individual experiment validation passed")
    
    except jsonschema.exceptions.ValidationError as e:
        print(f"individual experiment validation failed")

    experiments = composed_experiment["experiments"]
    techniques = [experiment["technique"] for experiment in experiments]

    # REM has to go first in the list of techniques
    if "rem" in techniques and techniques[0] != "rem":
        raise ValueError("REM is only compatible if applied as the first error mitigation technique.")



    # so now we have the ordered list of techniques the user is trying to use, so now we will check compatibility:
    # zne, pec, and ddd are the three we have right now:
    # -------> zne+pec: if zne is first, would want to apply pec to each of the noise scaled s=circuits, so would not be using execute_with_zne
    #                   if pec is first, would want to apply zne to each of the sampled circuits
    # -------> zne+ddd: if zne is first, would want to apply ddd to each of the noise scaled circuits
    #                   if ddd is first,would noise scale the circuit with the windows filled (not sure this makes sense to do?)
    # -------> pec+ddd: if pec is first, would want to apply ddd to each of the sampled circuits
    #                   if ddd is first, would want to apply pec to each circuits with slack windows filled
    # -------> zne+pec+ddd: if zne is first, would want to apply pec to each of the noise scaled circuits, then apply ddd to each of the pec circuits
    # REM needs and executor that returns raw measurment results
    


    return None


def basic_noise(n_qubits, prob=0.005):

    noise_model = NoiseModel()

    for i in range(n_qubits):
        readout_err = ReadoutError([[0.99, 0.01], [0.2, 0.8]])
        noise_model.add_readout_error(readout_err, [i])
        
    depolarizing_err1 = depolarizing_error(prob, num_qubits=1)
    depolarizing_err2 = depolarizing_error(prob, num_qubits=2)
    noise_model.add_all_qubit_quantum_error(depolarizing_err1, ['h', 'x', 'y', 'z'])
    noise_model.add_all_qubit_quantum_error(depolarizing_err2, ['cx'])

    return noise_model



# simple exector that returns the probability of measuring either |00...0>
#       eventually, will want an exector that does not actually run the circuits at all and instead computes overhead just from the mitigation experiment info

def execute_0s_ideal(circuit, noise_model=True):
    circ = circuit.copy()
    circ.measure_all()
    num_qubits = circ.num_qubits

    if noise_model:
        noise_model = basic_noise(num_qubits)
    
    backend = AerSimulator(noise_model=noise_model)
    transpiled_circuit = transpile(circ, optimization_level=0, backend=backend)

    results = backend.run(transpiled_circuit, optimization_level=0, shots=1000).result()
    counts = results.get_counts(transpiled_circuit)

    total_shots = sum(counts.values())
    probabilities = {state: count / total_shots for state, count in counts.items()}

    expectation_value = probabilities['0'*num_qubits]
    return expectation_value