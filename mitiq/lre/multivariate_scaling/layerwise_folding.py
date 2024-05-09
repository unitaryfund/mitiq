# Copyright (C) Unitary Fund
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

"""Functions for layerwise folding of input circuits to allow for multivariate
extrapolation."""

from copy import deepcopy

import cirq

from mitiq.utils import _pop_measurements
from mitiq.zne.scaling.folding import _check_foldable


def _get_num_layers_without_measurements(input_circuit: cirq.Circuit) -> int:
    """Checks if the circuit has non-terminal measurements and returns the
    number of layers in the input circuit without the terminal measurements.

        Args:
        circuit: Checks whether this circuit is able to be folded.

        Raises:
            UnfoldableCircuitError:
                * If the circuit has intermediate measurements.
                * If the circuit has non-unitary channels which are not
                terminal measurements.

        Returns:
            num_layers: the number of layers in the input circuit without the
            terminal measurements.
    """
    circuit = deepcopy(input_circuit)
    _check_foldable(circuit)
    _pop_measurements(circuit)
    num_layers = len(circuit)
    return num_layers
