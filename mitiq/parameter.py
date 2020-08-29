from typing import Iterable, Optional
import numpy as np

import copy

from cirq import Circuit, Gate, value, unitary, Moment, X, Z, Y
from cirq import (
    ZPowGate, YPowGate, XPowGate,
    HPowGate, CXPowGate, CZPowGate,
    MeasurementGate
)
from cirq.ops import gate_features
from cirq import protocols
from mitiq.folding import converter

BASE_GATES = [ZPowGate, HPowGate, XPowGate, YPowGate, CXPowGate, CZPowGate]


@converter
def scale_parameters(
    circ: Circuit,
    scale_factor: float,
    sigma: float,
    seed: Optional[int] = None
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
                base_gate = _get_base_gate(gate)
                param = gate.exponent * np.pi
                error = rng.normal(loc=0.0, scale=np.sqrt(noise))
                new_param = (param + error)
                curr_moment.append(
                    base_gate(exponent=new_param/np.pi)(*qubits))
        final_moments.append(Moment(curr_moment))
    return Circuit(final_moments)


def _get_base_gate(gate):
    for base_gate in BASE_GATES:
        if isinstance(gate, base_gate):
            return base_gate
    raise Exception(
        "Must have circuit be made of rotation gates. \
        Your gate {} may not be supported".format(gate))