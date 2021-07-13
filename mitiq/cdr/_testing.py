# Copyright (C) 2021 Unitary Fund
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

import numpy as np
import cirq

from mitiq import QPROGRAM
from mitiq.interface import accept_any_qprogram_as_input
from mitiq.cdr.execute import MeasurementResult


def random_x_z_circuit(qubits, n_moments, random_state) -> cirq.Circuit:
    angles = np.linspace(0.0, 2 * np.pi, 6)
    oneq_gates = [cirq.ops.rz(a) for a in angles] + [cirq.ops.rx(np.pi / 2)]
    gate_domain = {oneq_gate: 1 for oneq_gate in oneq_gates}

    return cirq.testing.random_circuit(
        qubits=qubits,
        n_moments=n_moments,
        op_density=1.0,
        gate_domain=gate_domain,
        random_state=random_state,
    )


def random_x_z_cnot_circuit(qubits, n_moments, random_state) -> cirq.Circuit:
    angles = np.linspace(0.0, 2 * np.pi, 8)
    oneq_gates = [
        gate(a) for a in angles for gate in (cirq.ops.rx, cirq.ops.rz)
    ]
    gate_domain = {oneq_gate: 1 for oneq_gate in oneq_gates}
    gate_domain.update({cirq.ops.CNOT: 2})

    return cirq.testing.random_circuit(
        qubits=qubits,
        n_moments=n_moments,
        op_density=1.0,
        gate_domain=gate_domain,
        random_state=random_state,
    )


@accept_any_qprogram_as_input
def executor(
    circuit: QPROGRAM, noise_level: float = 0.1, shots: int = 8192
) -> MeasurementResult:
    """Returns computational basis measurements after executing the circuit
    with depolarizing noise.

    Args:
        circuit: Circuit to execute.
        noise_level: Probability of depolarizing noise after each moment.
        shots: Number of samples to take.

    Returns:
        Dictionary where each key is a bitstring (binary int) and each value
        is the number of times that bitstring was measured.
    """
    circuit = circuit.with_noise(cirq.depolarize(p=noise_level))
    circuit.append(cirq.measure(*circuit.all_qubits(), key="z"))

    result = cirq.DensityMatrixSimulator().run(circuit, repetitions=shots)
    return result.histogram(key="z")


def simulator(circuit: QPROGRAM, shots: int = 8192) -> MeasurementResult:
    """Returns computational basis measurements after executing the circuit
    (without noise).

    Args:
        circuit: Circuit to simulate.
        shots: Number of samples to take.

    Returns:
        Dictionary where each key is a bitstring (binary int) and each value
        is the number of times that bitstring was measured.
    """
    return executor(circuit, noise_level=0.0, shots=shots)


@accept_any_qprogram_as_input
def simulator_statevector(circuit: QPROGRAM) -> np.ndarray:
    """Returns the final wavefunction (as a numpy array) of the circuit.

    Args:
        circuit: Circuit to simulate.
    """
    return cirq.Simulator().simulate(circuit).final_state_vector
