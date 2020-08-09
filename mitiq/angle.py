# needed imports
from typing import Iterable, Optional, Sequence, Tuple, Union, TYPE_CHECKING
import numpy as np

import matplotlib.pyplot as plt
import copy

from cirq import Circuit, DensityMatrixSimulator, Gate, value, unitary, Moment

from mitiq import execute_with_zne

from cirq import ops 
from cirq.ops import gate_features

from cirq import rx, ry, rz, CZ, X, Z, Y


from cirq import ZPowGate,YPowGate,XPowGate, HPowGate, CXPowGate, CZPowGate, MeasurementGate

def angle_noise_1q(noise, H):
    return AngleChannel1Q(noise, H)

def angle_noise_2q(noise, H):
    return AngleChannel2Q(noise, H)

def add_parameter_noise(circ: QPROGRAM, scale_factor: float, sigma: float) -> QPROGRAM:
    """Adds angle noise to a circuit with level noise.
    This adds noise to the actual parameter instead of 
    adding an angle channel.
    
    Args:
        circ: The quantum program as a cirq object. Assuming
        		last moment is a measurement moment.
        noise: the variance of rotations
    
    Returns:
        Noisy circuit with noise gates after each appropriate gate.
    
    """
    final_moments = []
    noise = (scale_factor - 1)*sigma
    for moment in circ:
        curr_moment = []
        for op in moment.operations:
            gate = copy.deepcopy(op.gate)
            qubits = op.qubits
            if isinstance(gate, MeasurementGate):
                curr_moment.append(gate(*qubits))
            else: 
                base_gate = _get_base_gate(gate)
                param = gate.exponent*np.pi
                error = np.random.normal(loc=0.0, scale=np.sqrt(noise))
                new_param = (param + error)
                curr_moment.append(base_gate(exponent = new_param/np.pi)(*qubits))
        final_moments.append(Moment(curr_moment))
    return Circuit(final_moments)

def _get_base_gate(gate):
    if isinstance(gate, ZPowGate):
        return ZPowGate
    if isinstance(gate, HPowGate):
        return HPowGate
    if isinstance(gate, XPowGate):
        return XPowGate
    if isinstance(gate, YPowGate):
        return YPowGate
    if isinstance(gate, CXPowGate):
        return CXPowGate
    if isinstance(gate, CZPowGate):
        return CZPowGate
    else:
        raise Exception("Must have circuit be made of rotation gates. Your gate {} may not be supported".format(gate))

def add_parameter_noise_channel(circ: Circuit, noise=None) -> Circuit:
    """Adds angle noise to a circuit with level noise via noise channel.
    
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
                noisy_op = angle_noise_1q(noise, gate)(qubits)
            else:
                noisy_op = angle_noise_2q(noise, gate)(qubits[0], qubits[1])
            curr_moment.append(noisy_op)
        final_moments.append(Moment(curr_moment))
    return Circuit(final_moments)

@value.value_equality
class AngleChannel1Q(gate_features.SingleQubitGate):
    """A channel that depolarizes a qubit."""

    def __init__(self, noise: float, H: Gate) -> None:
        r"""The Angle channel.

        This channel applies the angle sampling channel that is described
        in our literature. Given a noise level "noise", we are able to 
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
            self._H = X
        elif isinstance(H, YPowGate):
            self._H = Y
        elif isinstance(H, ZPowGate):
            self._H = Z
        else:
            self._H = H
        self.unitary = unitary(self._H)
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
        return 'cirq.angle_channel(noise={!r},H={!r})'.format(
            self._noise, self._H
        )

    def __str__(self) -> str:
        return 'angle_channel(noise={!r},H={!r})'.format(
            self._noise, self._H
        )

    def _circuit_diagram_info_(self,
                               args: 'protocols.CircuitDiagramInfoArgs') -> str:
        if args.precision is not None:
            f = '{:.' + str(args.precision) + 'g}'
            return 'AC({},{})'.format(f, self._H).format(self._noise)
        return 'AC({!r}, {})'.format(self._noise, self._H)

    def _json_dict_(self):
        return protocols.obj_to_dict_helper(self, ['noise'])

@value.value_equality
class AngleChannel2Q(gate_features.TwoQubitGate):
    """A channel that depolarizes a qubit."""

    def __init__(self, noise: float, H: Gate) -> None:
        r"""The Angle channel for two qubits

        This channel applies the angle sampling channel that is described
        in our literature. Given a noise level "noise", we are able to 
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
        self._H = H
        self.unitary = unitary(self._H)
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
        return 'cirq.angle_channel_2q(noise={!r},H={!r})'.format(
            self._noise, self._H
        )

    def __str__(self) -> str:
        return 'angle_channel_2q(noise={!r},H={!r})'.format(
            self._noise, self._H
        )

    def _circuit_diagram_info_(self,
                               args: 'protocols.CircuitDiagramInfoArgs') -> str:
        if args.precision is not None:
            f = '{:.' + str(args.precision) + 'g}'
            return ['AC2({},{})'.format(f, self._H).format(self._noise)]*2
        return ['AC2({!r}, {})'.format(self._noise, self._H)]*2

    def _json_dict_(self):
        return protocols.obj_to_dict_helper(self, ['noise'])