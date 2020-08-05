from typing import Optional
from qiskit import Aer, execute, QuantumCircuit

# Noise simulation packages
from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer.noise.errors.standard_errors import (
    depolarizing_error,
)

from mitiq.benchmarks.randomized_benchmarking import rb_circuits
from mitiq.mitiq_qiskit.conversions import to_qiskit

BACKEND = Aer.get_backend("qasm_simulator")


def random_one_qubit_identity_circuit(num_cliffords: int) -> QuantumCircuit:
    """Returns a single-qubit identity circuit.

    Args:
        num_cliffords (int): Number of cliffords used to generate the circuit.

    Returns:
        circuit: Quantum circuit as a :class:`qiskit.QuantumCircuit` object.
    """
    return to_qiskit(
        rb_circuits(n_qubits=1, num_cliffords=[num_cliffords], trials=1)[0]
    )


def run_with_noise(
    circuit: QuantumCircuit,
    noise: float,
    shots: int,
    seed: Optional[int] = None,
) -> float:
    """Runs the quantum circuit with a depolarizing channel noise model.

    Args:
        circuit: Ideal quantum circuit.
        noise: Noise constant going into `depolarizing_error`.
        shots: The Number of shots to run the circuit on the back-end.
        seed: Optional seed for qiskit simulator.

    Returns:
        expval: expected values.
    """
    # initialize a qiskit noise model
    noise_model = NoiseModel()

    # we assume a depolarizing error for each gate of the standard IBM basis
    # set (u1, u2, u3)
    noise_model.add_all_qubit_quantum_error(
        depolarizing_error(noise, 1), ["u1", "u2", "u3"]
    )

    # execution of the experiment
    job = execute(
        circuit,
        backend=BACKEND,
        basis_gates=["u1", "u2", "u3"],
        # we want all gates to be actually applied,
        # so we skip any circuit optimization
        optimization_level=0,
        noise_model=noise_model,
        shots=shots,
        seed_simulator=seed,
    )
    results = job.result()
    counts = results.get_counts()
    expval = counts["0"] / shots
    return expval


# For QISKIT the noise params are attributes of the simulation run and not of
# the program
# this means we need a stateful record of the scaled noise.
# Note this is NOT A GOOD SOLUTION IN THE LONG TERM AS HIDDEN STATE IS BAD
# Mainly this is qiskit's fault...
NATIVE_NOISE = 0.009
CURRENT_NOISE = None


def scale_noise(pq: QuantumCircuit, param: float) -> QuantumCircuit:
    """Scales the noise in a quantum circuit of the factor `param`.

    Args:
        pq: Quantum circuit.
        noise: Noise constant going into `depolarizing_error`.
        shots: Number of shots to run the circuit on the back-end.

    Returns:
        pq: quantum circuit as a :class:`qiskit.QuantumCircuit` object.
    """
    global CURRENT_NOISE
    noise = param * NATIVE_NOISE
    assert noise <= 1.0, (
        "Noise scaled to {} is out of bounds (<=1.0) for depolarizing "
        "channel.".format(noise)
    )

    noise_model = NoiseModel()
    # we assume a depolarizing error for each gate of the standard IBM basis
    # set (u1, u2, u3)
    noise_model.add_all_qubit_quantum_error(
        depolarizing_error(noise, 1), ["u1", "u2", "u3"]
    )
    CURRENT_NOISE = noise_model
    return pq


def run_program(
    pq: QuantumCircuit, shots: int = 100, seed: Optional[int] = None
) -> float:
    """Runs a single-qubit circuit for multiple shots and
    returns the expectation value of the ground state projector.


    Args:
        pq: Quantum circuit.
        shots: Number of shots to run the circuit on the back-end.
        seed: Optional seed for qiskit simulator.

    Returns:
        expval: expected value.
    """
    job = execute(
        pq,
        backend=BACKEND,
        basis_gates=["u1", "u2", "u3"],
        # we want all gates to be actually applied,
        # so we skip any circuit optimization
        optimization_level=0,
        noise_model=CURRENT_NOISE,
        shots=shots,
        seed_simulator=seed,
    )
    results = job.result()
    counts = results.get_counts()
    expval = counts["0"] / shots
    return expval
