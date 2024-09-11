"""Unit tests for the LRE extrapolation methods."""

from cirq import DensityMatrixSimulator, depolarize

from mitiq import benchmarks
from mitiq.lre import execute_with_lre

test_cirq = benchmarks.generate_rb_circuits(
    n_qubits=1,
    num_cliffords=2,
)[0]


def execute(circuit, noise_level=0.025, shots=1000):
    """Returns Tr[ρ |0⟩⟨0|] where ρ is the state prepared by the circuit
    executed with depolarizing noise.
    """
    # Replace with code based on your frontend and backend.
    mitiq_circuit = circuit
    noisy_circuit = mitiq_circuit.with_noise(depolarize(p=noise_level))
    rho = DensityMatrixSimulator().simulate(noisy_circuit).final_density_matrix
    return rho[0, 0].real


def test_lre_exp_value():
    noisy_val = execute(test_cirq)
    ideal_val = execute(test_cirq, noise_level=0, shots=1000)
    assert abs(ideal_val - noisy_val) > 0
    lre_exp_val = execute_with_lre(test_cirq, execute, 1000, 2, 2)
    assert lre_exp_val > noisy_val
