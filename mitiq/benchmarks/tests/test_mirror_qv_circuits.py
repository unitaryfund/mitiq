# Copyright (C) Unitary Fund
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

"""Tests for mirror quantum volume circuits."""

import pytest
import cirq

from mitiq.benchmarks.mirror_qv_circuits import generate_mirror_qv_circuit

def test_generate_mirror_qv_circuit_bitstring():
    test_circ, _ =  generate_mirror_qv_circuit(4, 3)
    bit_test= cirq.Simulator().run(test_circ, repetitions=1000)
    test_bitstring = list(bit_test.measurements.values())[0][0].tolist()
    expected_bitstring = [0] * 4
    assert test_bitstring == expected_bitstring




