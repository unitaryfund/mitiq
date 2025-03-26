from typing import Callable

import cirq
import numpy as np

from mitiq import MeasurementResult
from mitiq.vd.vd_utils import _apply_diagonalizing_gate, _copy_circuit_parallel


def execute_with_vd(
    circuit: cirq.Circuit,
    executor: Callable[[cirq.Circuit], MeasurementResult],
    num_shots: int = 1001, # TODO: this is needed for normalization
) -> list[float]:
    """Given a circuit that acts on N qubits, this function returns the
    expectation values of a given observable for each qubit i.
    The expectation values are corrected using the virtual distillation
    algorithm.

    Args:
        circuit: The input circuit of N qubits to execute with VD.
        executor: An executor that executes a circuit and returns either a
            density matrix, or a measurement result (bitstring).

    Returns:
        A list of VD-estimated expectation values for <Z_i>.
    """
    NUM_COPIES = 2
    num_qubits = len(circuit.all_qubits())
    parallel_copied_circuit = _copy_circuit_parallel(circuit, NUM_COPIES)
    entangled_circuit = _apply_diagonalizing_gate(
        parallel_copied_circuit, NUM_COPIES
    )
    entangled_circuit.append(cirq.measure(entangled_circuit.all_qubits()))

    results = executor(entangled_circuit)  # TODO: drop one shot if even?
    # print(results)
    subsystem1_bitstrings = results.filter_qubits(list(range(num_qubits)))
    subsystem2_bitstrings = results.filter_qubits(
        list(range(num_qubits, NUM_COPIES * num_qubits))
    )

    # Map 0 -> 1 and 1 -> -1
    z1 = 1 - 2 * subsystem1_bitstrings
    z2 = 1 - 2 * subsystem2_bitstrings

    expvals = []
    for i in range(num_qubits):
        expression = (z1[i] + z2[i]) / 2**num_qubits
        products = [
            1 + z1[j] - z2[j] + z1[j] * z2[j]
            for j in range(num_qubits)
            if j != i
        ]
        expvals.append(expression * np.prod(products))
    
    norm_products = [1 + z1[j] - z2[j] + z1[j] * z2[j] for j in range(num_qubits)]
    normalization = num_shots * np.prod(norm_products)
    return expvals / normalization
