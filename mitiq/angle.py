# needed imports
from typing import Iterable, Optional, Sequence, Tuple, Union, TYPE_CHECKING
import numpy as np

import matplotlib.pyplot as plt


from cirq import Circuit, DensityMatrixSimulator, Gate, value, unitary, Moment

from mitiq import execute_with_zne

from cirq import ops 
from cirq.ops import gate_features

@value.value_equality
class AngleChannel(gate_features.SingleQubitGate):
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
            return 'AC({},{})'.format(f, f).format(self._noise, self._H)
        return 'AC({!r},{!r})'.format(self._noise, self._H)

    def _circuit_diagram_info_(self,
                               args: 'protocols.CircuitDiagramInfoArgs') -> str:
        if args.precision is not None:
            f = '{:.' + str(args.precision) + 'g}'
            return 'AC({},{})'.format(f, self._H).format(self._noise)
        return 'AC({!r}, {})'.format(self._noise, self._H)

    def _json_dict_(self):
        return protocols.obj_to_dict_helper(self, ['noise'])
    
def angle_noise(noise, H):
    return AngleChannel(noise, H)

def add_angle_noise(circ: Circuit, noise=None) -> Circuit:
    """Adds angle noise to a circuit with level noise.
    
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
            noisy_op = angle_noise(noise, gate)(qubits)
            curr_moment.append(noisy_op)
        final_moments.append(Moment(curr_moment))

        #     gate = moment.operations[0].gate
        #     qubits = moment.operations[0].qubits
        #     if len(qubits) == 1:
        #         qubits = qubits[0]
        #     final_moments.append(moment)
        #     noisy_op = angle_noise(noise, gate)
        #     import pdb; pdb.set_trace()
        #     final_moments.append(noisy_op(qubits))
        # except:
        #     import pdb; pdb.set_trace()
    return Circuit(final_moments)