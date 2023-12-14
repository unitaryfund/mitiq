# Copyright (C) Unitary Fund
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

"""Tests for quantum volume circuits. The Cirq functions that do the main work
are tested here:
cirq-core/cirq/contrib/quantum_volume/quantum_volume_test.py

Tests below check that generate_quantum_volume_circuit() works as a wrapper and
fits with Mitiq's interface.
"""

import cirq
import pytest

from mitiq import SUPPORTED_PROGRAM_TYPES
from mitiq.benchmarks.randomized_clifford_t_circuit import (
    generate_random_clifford_t_circuit,
)


def test_generate_model_circuit_with_seed():
    """Test that a model circuit is determined by its seed."""
    circuit_1 = generate_random_clifford_t_circuit(
        num_qubits=2,
        num_oneq_cliffords=2,
        num_twoq_cliffords=2,
        num_t_gates=2,
        seed=1,
    )
    circuit_2 = generate_random_clifford_t_circuit(
        num_qubits=2,
        num_oneq_cliffords=2,
        num_twoq_cliffords=2,
        num_t_gates=2,
        seed=1,
    )
    circuit_3 = generate_random_clifford_t_circuit(
        num_qubits=2,
        num_oneq_cliffords=2,
        num_twoq_cliffords=2,
        num_t_gates=2,
        seed=3,
    )

    assert circuit_1 == circuit_2
    assert circuit_2 != circuit_3

@pytest.mark.parametrize("return_type", SUPPORTED_PROGRAM_TYPES.keys())
def test_conversion(return_type):
    circuit = generate_random_clifford_t_circuit(
        num_qubits=2,
        num_oneq_cliffords=2,
        num_twoq_cliffords=2,
        num_t_gates=2,
        return_type=return_type,
        seed=3,
    )
    assert return_type in circuit.__module__
