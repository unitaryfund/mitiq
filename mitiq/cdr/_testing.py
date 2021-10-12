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
