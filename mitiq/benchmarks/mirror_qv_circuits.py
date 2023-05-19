# Copyright (C) Unitary Fund
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

"""Functions to create a Mirror Quantum Volume Benchmarking circuit
as defined in https://arxiv.org/abs/2303.02108."""

from typing import Optional, Tuple, Sequence
import numpy as np

from mitiq import QPROGRAM
from mitiq.interface import convert_from_mitiq

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
    followed by an inverse of the quantum volume portion upto `depth/2` when `depth`
    is an even number. 
    
    When `depth` is odd, the layers will be chnaged to `depth+1`.
    

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
    
    if depth%2 !=0:
        depth = depth + 1
    else:
        depth = depth
    first_half_depth = int(depth/2)

    circ = cirq.Circuit()
    qv_half = generate_quantum_volume_circuit(num_qubits, first_half_depth, seed=seed, decompose=decompose)
    mirror_qv_half = cirq.inverse(qv_half[0])
    circ.append(qv_half[0], mirror_qv_half)
    
    #un-squash circuit moments
    output_circ = cirq.Circuit()
    output_ops = list(circ.all_operations())
    for i in output_ops:
        output_circ.append(i, strategy=cirq.InsertStrategy.NEW)
        
    
    # get the bitstring
    circ_with_measurements = output_circ + cirq.measure(output_circ.all_qubits())
    simulate_result = cirq.Simulator().run(circ_with_measurements)
    bitstring = list(simulate_result.measurements.values())[0][0].tolist()
    

    return(output_circ, bitstring)