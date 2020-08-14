from typing import Iterable
import numpy as np

import copy

from cirq import Circuit, Gate, value, unitary, Moment, X, Z, Y
from cirq import (
    ZPowGate, YPowGate, XPowGate,
    HPowGate, CXPowGate, CZPowGate,
    MeasurementGate
)
from cirq.ops import gate_features

from mitiq._typing import QPROGRAM
from mitiq.folding import converter

BASE_GATES = [ZPowGate, HPowGate, XPowGate, YPowGate, CXPowGate \
            CZPowGate]

@converter
def scale_parameters(
    circ: Circuit,
    scale_factor: float,
    sigma: float,
    seed: int = None
) -> Circuit:
    """Adds parameter noise to a circuit with level noise.
    This adds noise to the actual parameter instead of
    adding an parameter channel.

    Args:
        circ: The quantum program as a Cirq circuit object. All measurements
            should be in the last moment of the circuit.
        scale_factor: Amount to scale the base noise level of parameter rotations by.
        sigma: Base noise level (variance) in parameter rotations
        seed: 

    Returns:
        The input circuit with scaled rotation angles

    """
    final_moments = []
    noise = (scale_factor - 1) * sigma
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
                rng = np.random.RandomState(seed)
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


def scale_parameters_channel(circ: Circuit, noise=None) -> Circuit:
    """Adds parameter noise to a circuit with level noise via noise channel.

    Args:
        circ: The quantum program as a cirq object.
        noise: the base noise level to put into circuit (variance of rotations)

    Returns:
        Noisy circuit with noise gates after each appropriate gate.

    """
    final_moments = []
    for moment in circ:
        final_moments.append(moment)
        curr_moment = []
        for op in moment.operations:
            gate = op.gate
            qubits = op.qubits
            if len(qubits) == 1:
                qubits = qubits[0]
                noisy_op = parameter_noise_1q(noise, gate)(qubits)
            elif len(qubits) == 2:
                noisy_op = parameter_noise_2q(noise, gate)(qubits[0], qubits[1])
            else:
                raise Exception(
                    "Gates for more than two qubits are not \
                    supported with parameter noise scaling.")
            curr_moment.append(noisy_op)
        final_moments.append(Moment(curr_moment))
    return Circuit(final_moments)


def parameter_noise_1q(noise, base_gate):
    return parameterChannel1Q(noise, base_gate)


def parameter_noise_2q(noise, base_gate):
    return parameterChannel2Q(noise, base_gate)


@value.value_equality
class parameterChannel1Q(gate_features.SingleQubitGate):
    """A channel that depolarizes a qubit."""

    def __init__(self, noise: float, base_gate: Gate) -> None:
        r"""The parameter channel.

        This channel applies the parameter sampling channel that is described
        in our literature (https://arxiv.org/abs/2005.10921).
        Given a noise level "noise", we are able to
        compute what the channel is:

        This channel evolves a density matrix via

            $$
            \rho \rightarrow sqrt(1 - Q) \rho
                    + sqrt(Q) H \rho H
            $$

        Args:
           noise: the variance of the gate.
           H: the gate to be applied in event of error.

        Raises:
            ValueError: if p is not a valid probability.
        """

        self._noise = noise

        if isinstance(H, XPowGate):
            self._base_gate = X
        elif isinstance(H, YPowGate):
            self._base_gate = Y
        elif isinstance(H, ZPowGate):
            self._base_gate = Z
        else:
            self._base_gate = H
        self.unitary = unitary(self._base_gate)
        self._Q = self.calc_Q()

    def calc_Q(self):
        return 0.5*(1-np.exp(-2*self._noise))

    def _channel_(self) -> Iterable[np.ndarray]:
        return (
            np.sqrt(1-self._Q) * np.array([[1., 0.], [0., 1.]]),
            np.sqrt(self._Q) * self.unitary,
        )

    def _has_channel_(self) -> bool:
        return True

    def _value_equality_values_(self):
        return self._noise

    def __repr__(self) -> str:
        return 'cirq.parameter_channel(noise={!r},H={!r})'.format(
            self._noise, self._base_gate
        )

    def __str__(self) -> str:
        return 'parameter_channel(noise={!r},H={!r})'.format(
            self._noise, self._base_gate
        )

    def _circuit_diagram_info_(
        self,
        args: 'protocols.CircuitDiagramInfoArgs'
    ) -> str:
        if args.precision is not None:
            f = '{:.' + str(args.precision) + 'g}'
            return 'AC({},{})'.format(f, self._base_gate).format(self._noise)
        return 'AC({!r}, {})'.format(self._noise, self._base_gate)

    def _json_dict_(self):
        return protocols.obj_to_dict_helper(self, ['noise'])


@value.value_equality
class parameterChannel2Q(gate_features.TwoQubitGate):
    """A channel that depolarizes a qubit."""

    def __init__(self, noise: float, base_gate: Gate) -> None:
        r"""The parameter channel for two qubits

        This channel applies the parameter sampling channel that is described
        in our literature (https://arxiv.org/abs/2005.10921).
        Given a noise level "noise", we are able to
        compute what the channel is:

        This channel evolves a density matrix via

            $$
            \rho \rightarrow sqrt(1 - Q) \rho
                    + sqrt(Q) H \rho H
            $$

        Args:
           noise: the variance of the gate.
           H: the gate to be applied in event of error.

        Raises:
            ValueError: if p is not a valid probability.
        """

        self._noise = noise
        self._base_gate = base_gate
        self.unitary = unitary(self._base_gate)
        self._Q = self.calc_Q()

    def calc_Q(self):
        return 0.5*(1-np.exp(-2*self._noise))

    def _channel_(self) -> Iterable[np.ndarray]:
        return (
            np.sqrt(1-self._Q) * np.eye(4),
            np.sqrt(self._Q) * self.unitary,
        )

    def _has_channel_(self) -> bool:
        return True

    def _value_equality_values_(self):
        return self._noise

    def __repr__(self) -> str:
        return 'cirq.parameter_channel_2q(noise={!r},H={!r})'.format(
            self._noise, self._base_gate
        )

    def __str__(self) -> str:
        return 'parameter_channel_2q(noise={!r},H={!r})'.format(
            self._noise, self._base_gate
        )

    def _circuit_diagram_info_(
        self,
        args: 'protocols.CircuitDiagramInfoArgs'
    ) -> str:
        if args.precision is not None:
            f = '{:.' + str(args.precision) + 'g}'
            return ['AC2({},{})'.format(f, self._base_gate).format(self._noise)]*2
        return ['AC2({!r}, {})'.format(self._noise, self._base_gate)]*2

    def _json_dict_(self):
        return protocols.obj_to_dict_helper(self, ['noise'])
