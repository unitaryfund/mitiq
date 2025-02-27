"""Unit tests for the LRE extrapolation methods."""

import math
import random
import re
from typing import List
from unittest.mock import Mock

import pytest
from cirq import DensityMatrixSimulator, depolarize

from mitiq import SUPPORTED_PROGRAM_TYPES, Executor, benchmarks
from mitiq.lre import (
    execute_with_lre,
    lre_decorator,
)
from mitiq.lre.multivariate_scaling.layerwise_folding import (
    _get_chunks,
    multivariate_layer_scaling,
)
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


def batched_executor(circuits) -> List[float]:
    return [execute(circuit) for circuit in circuits]


noisy_val = execute(test_cirq)
ideal_val = execute(test_cirq, noise_level=0)


@pytest.mark.parametrize("degree, fold_multiplier", [(2, 2), (2, 3), (3, 4)])
def test_lre_exp_value(degree, fold_multiplier):
    """Verify LRE executors work as expected."""
    lre_exp_val = execute_with_lre(
        test_cirq,
        execute,
        degree=degree,
        fold_multiplier=fold_multiplier,
    )
    assert abs(lre_exp_val - ideal_val) <= abs(noisy_val - ideal_val)


@pytest.mark.parametrize(
    "degree, fold_multiplier",
    [(2, 2), (2, 3), (3, 4)],
)
def test_lre_executor(degree, fold_multiplier):
    """Verify LRE batch executor works as expected."""
    test_executor = Executor(execute)
    lre_exp_val = execute_with_lre(
        test_cirq,
        test_executor,
        degree=degree,
        fold_multiplier=fold_multiplier,
    )
    assert isinstance(lre_exp_val, float)

    assert test_executor.calls_to_executor == len(
        multivariate_layer_scaling(test_cirq, degree, fold_multiplier)
    )


@pytest.mark.parametrize(
    "degree, fold_multiplier",
    [(2, 2), (2, 3), (3, 4)],
)
def test_lre_batched_executor(degree, fold_multiplier):
    """Verify LRE batch executor works as expected."""
    test_batched_executor = Executor(batched_executor, max_batch_size=200)
    lre_exp_val_batched = execute_with_lre(
        test_cirq,
        test_batched_executor,
        degree=degree,
        fold_multiplier=fold_multiplier,
    )
    assert isinstance(lre_exp_val_batched, float)

    assert test_batched_executor.calls_to_executor == 1
    assert (
        test_batched_executor.executed_circuits
        == multivariate_layer_scaling(test_cirq, degree, fold_multiplier)
    )


@pytest.mark.parametrize("circuit_type", SUPPORTED_PROGRAM_TYPES.keys())
def test_lre_all_qprogram(circuit_type):
    """Verify LRE works with all supported frontends."""
    degree, fold_multiplier = 2, 3
    circuit = benchmarks.generate_ghz_circuit(3, circuit_type)
    depth = 3  # not all circuit types have a simple way to compute depth

    mock_executor = Mock(side_effect=lambda _: random.random())

    lre_exp_val = execute_with_lre(
        circuit,
        mock_executor,
        degree=degree,
        fold_multiplier=fold_multiplier,
    )

    assert isinstance(lre_exp_val, float)
    assert mock_executor.call_count == math.comb(degree + depth, degree)


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


def test_lre_runs_correct_number_of_circuits_when_chunking():
    """Verify execute_with_lre works as expected when chunking is used.
    Note that this does not validate performance of chunking."""

    mock_executor = Mock(side_effect=lambda _: random.random())

    test_cirq = benchmarks.generate_rb_circuits(n_qubits=1, num_cliffords=12)[
        0
    ]

    degree, fold_multiplier, num_chunks = 2, 2, 10

    lre_exp_val_chunking = execute_with_lre(
        test_cirq,
        mock_executor,
        degree=degree,
        fold_multiplier=fold_multiplier,
        num_chunks=num_chunks,
    )

    chunked_circ = _get_chunks(test_cirq, num_chunks=num_chunks)
    assert isinstance(lre_exp_val_chunking, float)
    assert mock_executor.call_count == math.comb(
        degree + len(chunked_circ), degree
    )
