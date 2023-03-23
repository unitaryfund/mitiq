"""Tests for GHZ circuits."""
import pytest
import numpy as np
from mitiq.benchmarks import ghz_circuits
from mitiq import SUPPORTED_PROGRAM_TYPES

@pytest.mark.parametrize('nqubits', [1, 5])
def test_ghz_circuits(nqubits):
    circuit = ghz_circuits.generate_ghz_circuit(nqubits)
    sv = circuit.final_state_vector()
    zero_prob = abs(sv[0]) ** 2
    one_prob = abs(sv[-1]) ** 2
    assert np.isclose(zero_prob, 0.5)
    assert np.isclose(one_prob, 0.5)

@pytest.mark.order(0)
def test_ghz_value_error():
    with pytest.raises(ValueError):
        circuit = ghz_circuits.generate_ghz_circuit(0)

@pytest.mark.parametrize('return_type', SUPPORTED_PROGRAM_TYPES.keys())
def test_ghz_conversion(return_type):
    circuit = ghz_circuits.generate_ghz_circuit(3, return_type)
    assert return_type in circuit.__module__