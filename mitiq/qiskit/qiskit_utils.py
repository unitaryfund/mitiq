import numpy as np

import qiskit
from qiskit import QuantumCircuit

# Noise simulation packages
from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer.noise.errors.standard_errors import depolarizing_error

BACKEND = qiskit.Aer.get_backend('qasm_simulator')
# Set the random seeds for testing
QISKIT_SEED = 1337
np.random.seed(1001)


def random_identity_circuit(depth=None):
    """Returns a single-qubit identity circuit based on Pauli gates."""

    # initialize a quantum circuit with 1 qubit and 1 classical bit
    circuit = QuantumCircuit(1, 1)

    # index of the (inverting) final gate: 0=I, 1=X, 2=Y, 3=Z
    k_inv = 0

    # apply a random sequence of Pauli gates
    for _ in range(depth):
        # random index for the next gate: 1=X, 2=Y, 3=Z
        k = np.random.choice([1, 2, 3])
        # apply the Pauli gate "k"
        if k == 1:
            circuit.x(0)
        elif k == 2:
            circuit.y(0)
        elif k == 3:
            circuit.z(0)

        # update the inverse index according to
        # the product rules of Pauli matrices k and k_inv
        if k_inv == 0:
            k_inv = k
        elif k_inv == k:
            k_inv = 0
        else:
            _ = [1, 2, 3]
            _.remove(k_inv)
            _.remove(k)
            k_inv = _[0]

    # apply the final inverse gate
    if k_inv == 1:
        circuit.x(0)
    elif k_inv == 2:
        circuit.y(0)
    elif k_inv == 3:
        circuit.z(0)

    return circuit


def run_with_noise(circuit, noise, shots):
    # initialize a qiskit noise model
    noise_model = NoiseModel()

    # we assume a depolarizing error for each gate of the standard IBM basis set (u1, u2, u3)
    noise_model.add_all_qubit_quantum_error(depolarizing_error(noise, 1), ['u1', 'u2', 'u3'])

    # execution of the experiment
    job = qiskit.execute(circuit,
                         backend=BACKEND,
                         basis_gates=['u1', 'u2', 'u3'],
                         # we want all gates to be actually applied,
                         # so we skip any circuit optimization
                         optimization_level=0,
                         noise_model=noise_model,
                         shots=shots,
                         seed_simulator=QISKIT_SEED)
    results = job.result()
    counts = results.get_counts()
    expval = counts['0'] / shots
    return expval


# For QISKIT the noise params are attributes of the simulation run and not of the program
# this means we need a stateful record of the scaled noise.
# Note this is NOT A GOOD SOLUTION IN THE LONG TERM AS HIDDEN STATE IS BAD
# Mainly this is qiskit's fault...
NATIVE_NOISE = 0.009
CURRENT_NOISE = None


def scale_noise(pq, param: float):
    global CURRENT_NOISE
    noise = param * NATIVE_NOISE
    assert noise <= 1.0, "Noise scaled to {} is out of bounds (<=1.0) for depolarizing channel.".format(noise)

    noise_model = NoiseModel()
    # we assume a depolarizing error for each gate of the standard IBM basis set (u1, u2, u3)
    noise_model.add_all_qubit_quantum_error(depolarizing_error(noise, 1), ['u1', 'u2', 'u3'])
    CURRENT_NOISE = noise_model
    return pq


def run_program(pq, shots: int = 100) -> float:
    job = qiskit.execute(pq,
                         backend=BACKEND,
                         basis_gates=['u1', 'u2', 'u3'],
                         # we want all gates to be actually applied,
                         # so we skip any circuit optimization
                         optimization_level=0,
                         noise_model=CURRENT_NOISE,
                         shots=shots,
                         seed_simulator=QISKIT_SEED)
    results = job.result()
    counts = results.get_counts()
    expval = counts['0'] / shots
    return expval

def measure(circuit, qid):
    circuit.measure(0, qid)
    return circuit

