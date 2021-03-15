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

from typing import Optional, Callable, Iterable
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
    Gate,
    Qid,
)
from mitiq.conversions import converter


class GateTypeException(Exception):
    pass


def _get_base_gate(gate: EigenGate) -> EigenGate:
    BASE_GATES = [ZPowGate, HPowGate, XPowGate, YPowGate, CXPowGate, CZPowGate]

    for base_gate in BASE_GATES:
        if isinstance(gate, base_gate):
            return base_gate
    raise GateTypeException(
        "Must have circuit be made of rotation gates. "
        "Your gate {} may not be supported".format(gate)
    )


class CircuitMismatchException(Exception):
    pass


def _generate_parameter_calibration_circuit(
    qubits: Iterable, depth: int, gate: EigenGate
) -> Circuit:
    """
    Generates a circuit which should be the identity. Given a rotation
    gate R(param), it applies R(2 * pi / depth) depth times, resulting
    in R(2*pi). Requires that the gate is periodic in 2*pi.

    Args:
        qubits: a list of qubits
        depth: the length of the circuit to create
        gate: the base gate to apply several times, must be periodic
                in 2*pi

    Returns:
        circuit: a parameter calibration circuit that can be
                used for profiling
    """
    num_qubits = gate().num_qubits()
    if num_qubits != len(qubits):
        raise CircuitMismatchException(
            "Number of qubits does not match domain size of gate."
        )
    return Circuit(
        gate(exponent=2 * np.pi / depth).on(*qubits) for _ in range(depth)
    )


def _parameter_calibration(
    executor: Callable[..., float], gate: Gate, qubit: Qid, depth: int = 100
) -> float:
    """
    Given an executor and a gate, determines the effective
    variance in the control parameter
    that can be used for parameter noise scaling later on.
    Only works for one qubit gates for now.

    Args:
        executor: a function that takes in a quantum circuit and returns
            an expectation value
        gate: the quantum gate that you wish to profile
        qubit: the index of the qubit you wish to profile
        depth: the number of operations you would like to use to profile
            your gate.

    Returns:
        sigma: a float representing the standard deviation of the error
            of your gate
    """

    base_gate = _get_base_gate(gate)
    circuit = _generate_parameter_calibration_circuit(
        [qubit], depth, base_gate
    )
    expectation = executor(circuit)
    Q = (1 - np.power(2 * expectation - 1, 1 / depth)) / 2
    sigma = -0.5 * np.log(1 - 2 * Q)
    return sigma


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
