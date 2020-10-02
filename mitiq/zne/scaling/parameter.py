# Copyright (C) 2020 Unitary Fund
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

from typing import cast, Optional
import numpy as np

import copy

from cirq import Circuit, EigenGate, Moment
from cirq import (
    ZPowGate,
    YPowGate,
    XPowGate,
    HPowGate,
    CXPowGate,
    CZPowGate,
    MeasurementGate,
)
from mitiq.conversions import converter


BASE_GATES = [ZPowGate, HPowGate, XPowGate, YPowGate, CXPowGate, CZPowGate]


class GateTypeException(Exception):
    pass


def _get_base_gate(gate: EigenGate) -> EigenGate:
    for base_gate in BASE_GATES:
        if isinstance(gate, base_gate):
            return cast(EigenGate, base_gate)
    raise GateTypeException(
        "Must have circuit be made of rotation gates. "
        "Your gate {} may not be supported".format(gate)
    )


@converter
def scale_parameters(
    circ: Circuit,
    scale_factor: float,
    sigma: float,
    seed: Optional[int] = None,
) -> Circuit:
    """Adds parameter noise to a circuit with level noise.
    This adds noise to the actual parameter instead of
    adding an parameter channel.

    Args:
        circ: The quantum program as a Cirq circuit object. All measurements
            should be in the last moment of the circuit.
        scale_factor: Amount to scale the base noise level of parameters by.
        sigma: Base noise level (variance) in parameter rotations
        seed: random seed

    Returns:
        The input circuit with scaled rotation angles

    """
    final_moments = []
    noise = (scale_factor - 1) * sigma
    rng = np.random.RandomState(seed)
    for moment in circ:
        curr_moment = []
        for op in moment.operations:
            gate = copy.deepcopy(op.gate)
            qubits = op.qubits
            if isinstance(gate, MeasurementGate):
                curr_moment.append(gate(*qubits))
            else:
                assert isinstance(gate, EigenGate)
                base_gate = _get_base_gate(gate)
                param = gate.exponent * np.pi
                error = rng.normal(loc=0.0, scale=np.sqrt(noise))
                new_param = param + error
                curr_moment.append(
                    base_gate(exponent=new_param / np.pi)(*qubits)
                )
        final_moments.append(Moment(curr_moment))
    return Circuit(final_moments)
