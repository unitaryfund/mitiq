"""Unit tests for the LRE extrapolation methods."""

import re

import pytest
from cirq import DensityMatrixSimulator, depolarize

from mitiq import benchmarks
from mitiq.lre import execute_with_lre, lre_decorator, mitigate_executor
from mitiq.zne.scaling import fold_all, fold_global

# default circuit for all unit tests
test_cirq = benchmarks.generate_rb_circuits(
    n_qubits=1,
    num_cliffords=2,
)[0]


# default execute function for all unit tests
def execute(circuit, noise_level=0.025):
    """Default executor for all unit tests."""
    noisy_circuit = circuit.with_noise(depolarize(p=noise_level))
    rho = DensityMatrixSimulator().simulate(noisy_circuit).final_density_matrix
    return rho[0, 0].real


noisy_val = execute(test_cirq)
ideal_val = execute(test_cirq, noise_level=0)


@pytest.mark.parametrize(
    "input_degree, input_fold_multiplier", [(2, 2), (2, 3), (3, 4)]
)
def test_lre_exp_value(input_degree, input_fold_multiplier):
    """Verify LRE executors work as expected."""
    lre_exp_val = execute_with_lre(
        test_cirq,
        execute,
        degree=input_degree,
        fold_multiplier=input_fold_multiplier,
    )
    assert abs(lre_exp_val - ideal_val) <= abs(noisy_val - ideal_val)


@pytest.mark.parametrize(
    "input_degree, input_fold_multiplier", [(2, 2), (2, 3), (3, 4)]
)
def test_lre_exp_value_decorator(input_degree, input_fold_multiplier):
    """Verify LRE mitigated executor work as expected."""
    mitigated_executor = mitigate_executor(
        execute, degree=2, fold_multiplier=2
    )
    exp_val_from_mitigate_executor = mitigated_executor(test_cirq)
    assert abs(exp_val_from_mitigate_executor - ideal_val) <= abs(
        noisy_val - ideal_val
    )


def test_lre_decorator():
    """Verify LRE decorators work as expected."""

    @lre_decorator(degree=2, fold_multiplier=2)
    def execute(circuit, noise_level=0.025):
        noisy_circuit = circuit.with_noise(depolarize(p=noise_level))
        rho = (
            DensityMatrixSimulator()
            .simulate(noisy_circuit)
            .final_density_matrix
        )
        return rho[0, 0].real

    assert abs(execute(test_cirq) - ideal_val) <= abs(noisy_val - ideal_val)


def test_lre_decorator_raised_error():
    """Verify an error is raised when the required parameters for the decorator
    are not specified."""
    with pytest.raises(TypeError, match=re.escape("lre_decorator() missing")):

        @lre_decorator()
        def execute(circuit, noise_level=0.025):
            noisy_circuit = circuit.with_noise(depolarize(p=noise_level))
            rho = (
                DensityMatrixSimulator()
                .simulate(noisy_circuit)
                .final_density_matrix
            )
            return rho[0, 0].real

        assert abs(execute(test_cirq) - ideal_val) <= abs(
            noisy_val - ideal_val
        )


def test_lre_executor_with_chunking():
    """Verify the executor works as expected for chunking a large circuit into
    a smaller circuit."""
    # define a larger circuit
    test_cirq = benchmarks.generate_rb_circuits(n_qubits=1, num_cliffords=12)[
        0
    ]
    lre_exp_val = execute_with_lre(
        test_cirq, execute, degree=2, fold_multiplier=2, num_chunks=14
    )
    assert abs(lre_exp_val - ideal_val) <= abs(noisy_val - ideal_val)


@pytest.mark.parametrize(
    "test_input", [(1), (2), (3), (4), (5), (6), (7), (8), (9)]
)
@pytest.mark.xfail
def test_lre_executor_with_chunking_failures(test_input):
    """Verify chunking fails when a large number of layers are chunked into a
    smaller number of circuit chunks."""
    # define a larger circuit
    test_cirq = benchmarks.generate_rb_circuits(n_qubits=1, num_cliffords=15)[
        0
    ]
    lre_exp_val = execute_with_lre(
        test_cirq, execute, degree=2, fold_multiplier=2, num_chunks=test_input
    )
    assert abs(lre_exp_val - ideal_val) <= abs(noisy_val - ideal_val)


@pytest.mark.parametrize("input_method", [(fold_global), (fold_all)])
def test_lre_executor_with_different_folding_methods(input_method):
    """Verify the executor works as expected for using non-default unitary
    folding methods."""
    lre_exp_val = execute_with_lre(
        test_cirq,
        execute,
        degree=2,
        fold_multiplier=2,
        folding_method=input_method,
    )
    assert abs(lre_exp_val - ideal_val) <= abs(noisy_val - ideal_val)
