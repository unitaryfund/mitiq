# Copyright (C) Unitary Fund
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

"""Tests for mirror quantum volume circuits."""


import cirq
import pytest

from mitiq.benchmarks.mirror_qv_circuits import generate_mirror_qv_circuit


@pytest.mark.parametrize(
    "depth_num",
    [1, 2, 3, 4, 5, 6, 7, 8],
)
def test_generate_mirror_qv_circuit(depth_num):
    test_circ, _ = generate_mirror_qv_circuit(4, depth_num)

    # check bitstring is all 0's
    bit_test = cirq.Simulator().run(
        test_circ + cirq.measure(test_circ.all_qubits()), repetitions=1000
    )
    test_bitstring = list(bit_test.measurements.values())[0][0].tolist()
    expected_bitstring = [0] * 4
    assert test_bitstring == expected_bitstring


def test_bad_depth_number():
    for n in (-1, 0):
        with pytest.raises(
            ValueError, match="{} is invalid for the generated circuit depth."
        ):
            generate_mirror_qv_circuit(3, n)
