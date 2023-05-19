# Copyright (C) 2023 Unitary Fund
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

"""Functions to create a Mirror Quantum Volume Benchmarking circuit
as defined in https://arxiv.org/abs/2303.02108."""

from mitiq.benchmarks.quantum_volume_circuits import generate_quantum_volume_circuit
import cirq

def generate_mirror_qv_circuit(
    num_qubits: int,
    depth: int,
    decompose: bool = False,
    seed: Optional[int] = None,
    return_type: Optional[str] = None,
) -> Tuple[QPROGRAM, Sequence[Bitstring]]:
    """Generate a mirror quantum volume circuit with the given number of qubits and
    depth.

    The generated circuit consists of a quantum volume circuit upto `depth/2` layers
    followed by an inverse of the quantum volume portion upto `depth/2`.

    The output bit-string is always supposed to be a string of zeroes. 

    Args:
        num_qubits: The number of qubits in the generated circuit.
        depth: The number of layers in the generated circuit.
        decompose: Recursively decomposes the randomly sampled (numerical)
            unitary matrix gates into simpler gates.
        seed: Seed for generating random circuit.
        return_type: String which specifies the type of the returned
            circuits. See the keys of ``mitiq.SUPPORTED_PROGRAM_TYPES``
            for options. If ``None``, the returned circuits have type
            ``cirq.Circuit``.

    Returns:
        A quantum volume circuit acting on ``num_qubits`` qubits.
        A list of the bitstrings for the returned circuit.
    """
    first_half_depth = int(depth/2)
    random_option = random.RandomState(seed)

    circ = cirq.Circuit()
    qv_half = generate_quantum_volume_circuit(num_qubits, first_half_depth, decompose, random_option)
    mirror_qv_half = cirq.inverse(qv_half)
    circ.append(qv_half, mirror_qv_half)

    return(circ)




