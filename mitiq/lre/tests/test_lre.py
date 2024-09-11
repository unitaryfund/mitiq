"""Unit tests for the LRE extrapolation methods."""

import re

import pytest
from cirq import DensityMatrixSimulator, depolarize

from mitiq import benchmarks
from mitiq.lre import execute_with_lre, lre_decorator, mitigate_executor

test_cirq = benchmarks.generate_rb_circuits(
    n_qubits=1,
    num_cliffords=2,
)[0]


def execute(circuit, noise_level=0.025, shots=1000):
    # Replace with code based on your frontend and backend.
    mitiq_circuit = circuit
    noisy_circuit = mitiq_circuit.with_noise(depolarize(p=noise_level))
    rho = DensityMatrixSimulator().simulate(noisy_circuit).final_density_matrix
    return rho[0, 0].real


noisy_val = execute(test_cirq)


def test_lre_exp_value():
    ideal_val = execute(test_cirq, noise_level=0, shots=1000)
    assert abs(ideal_val - noisy_val) > 0
    lre_exp_val = execute_with_lre(test_cirq, execute, 1000, 2, 2)
    assert lre_exp_val > noisy_val

    # verify the mitigated decorator work as expected
    mitigated_executor = mitigate_executor(execute, 1000, 2, 2)
    exp_val_from_mitigate_executor = mitigated_executor(test_cirq)
    assert exp_val_from_mitigate_executor > noisy_val


def test_lre_decorator():
    @lre_decorator(100, 2, 2)
    def execute(circuit, noise_level=0.025, shots=100):
        # Replace with code based on your frontend and backend.
        mitiq_circuit = circuit
        noisy_circuit = mitiq_circuit.with_noise(depolarize(p=noise_level))
        rho = (
            DensityMatrixSimulator()
            .simulate(noisy_circuit)
            .final_density_matrix
        )
        return rho[0, 0].real

    assert noisy_val < execute(test_cirq)


def test_lre_decorator_raised_error():
    with pytest.raises(TypeError, match=re.escape("lre_decorator() missing")):

        @lre_decorator()
        def execute(circuit, noise_level=0.025, shots=100):
            # Replace with code based on your frontend and backend.
            mitiq_circuit = circuit
            noisy_circuit = mitiq_circuit.with_noise(depolarize(p=noise_level))
            rho = (
                DensityMatrixSimulator()
                .simulate(noisy_circuit)
                .final_density_matrix
            )
            return rho[0, 0].real

        assert noisy_val < execute(test_cirq)
