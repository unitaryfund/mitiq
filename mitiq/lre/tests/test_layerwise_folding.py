# Copyright (C) Unitary Fund
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for scaling noise by unitary folding of layers in the input
circuit to allow for multivariate extrapolation."""

from copy import deepcopy

import cirq
import pytest

from mitiq.lre.multivariate_scaling.layerwise_folding import (
    _get_num_layers_without_measurements,
)

qreg1 = cirq.LineQubit.range(3)
test_circuit1 = cirq.Circuit(
    [cirq.ops.H.on_each(*qreg1)],
    [cirq.ops.CNOT.on(qreg1[0], qreg1[1])],
    [cirq.ops.X.on(qreg1[2])],
    [cirq.ops.TOFFOLI.on(*qreg1)],
)

test_circuit1_with_measurements = deepcopy(test_circuit1)
test_circuit1_with_measurements.append(cirq.ops.measure_each(*qreg1))


@pytest.mark.parametrize(
    "test_input, expected",
    [(test_circuit1, 3), (test_circuit1_with_measurements, 3)],
)
def test_get_num_layers(test_input, expected):
    """Verifies function works as expected."""
    calculated_num_layers = _get_num_layers_without_measurements(test_input)

    assert calculated_num_layers == expected
