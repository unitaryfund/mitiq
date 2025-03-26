import warnings
from typing import Callable, List

import cirq
import numpy as np

from mitiq import (
    Executor,
    MeasurementResult,
    Observable,
)
from mitiq.executor.executor import DensityMatrixLike, MeasurementResultLike
from mitiq.vd.vd_utils import (
    _apply_cyclic_system_permutation,
    _apply_diagonalizing_gate,
    _apply_symmetric_observable,
    _copy_circuit_parallel,
)


def vd_executor(
    circuit: cirq.Circuit, reps: int = 10
) -> List[MeasurementResult]:
    results = cirq.sample(circuit, repetitions=reps).measurements
    measurements = []

    sorted_keys = []
    for key in results:
        sorted_keys.append(int(key))
    sorted_keys.sort()

    for key in sorted_keys:
        measurements.append(results[str(key)])

    measurements = np.squeeze(measurements, axis=2).T

    return measurements.tolist()


def execute_with_vd(
    circuit: cirq.Circuit,
    executor: Callable[[cirq.Circuit], MeasurementResult],
    M: int = 2,  # TODO: refactor with num_additional_qubits
    observable: Observable | None = None,
) -> list[float]:
    """Given a circuit that acts on N qubits, this function returns the
    expectation values of a given observable for each qubit i.
    The expectation values are corrected using the virtual distillation
    algorithm.

    Args:
        circuit: The input circuit of N qubits to execute with VD.
        executor: An executor that executes a circuit and returns either a
            density matrix, or a measurement result (bitstring).
        M: The number of copies of rho. Only M=2 is implemented at this moment.
        observable: The one qubit observable for which the expectation values
            are computed. The default observable is the Pauli Z matrix. At the
            moment using different observables is not supported.

    Returns:
        A list of VD-estimated expectation values for <Z_i>.
    """

    num_qubits = len(circuit.all_qubits())
    parallel_copied_circuit = _copy_circuit_parallel(circuit, M)


    executor_obj = Executor(executor)  # type: ignore[arg-type]

    # Density matrix return type
    if executor_obj._executor_return_type in DensityMatrixLike:
        circuit_dm = executor_obj.run(parallel_copied_circuit)
        circuit_swaps = _apply_cyclic_system_permutation(
            circuit_dm, num_qubits
        )
        resulting_dm = _apply_symmetric_observable(
            circuit_swaps, num_qubits, observable
        )
        expectation_values = np.trace(resulting_dm, axis1=1, axis2=2) / np.trace(
            resulting_dm, axis1=1, axis2=2
        )

    elif executor_obj._executor_return_type in MeasurementResultLike:
        # TODO: why no cyclic permutation here?
        entangled_circuits = _apply_diagonalizing_gate(
            parallel_copied_circuit, M
        )

        entangled_circuits.append(
            cirq.measure(entangled_circuits.all_qubits())
        )

        meas_res = executor_obj.run(entangled_circuits, force_run_all=True)[0]
        # TODO: drop one shot if not even?

        Ei = np.zeros(num_qubits)
        D = 0
        # post processing measurements
        for bitstring in meas_res.asarray:
            # map 0/1 measurements to 1/-1 measurements (eigenvalues of Z)
            measurement_eigenvalues = np.array(
                [1 if bit == 0 else -1 for bit in bitstring]
            )
            z1 = measurement_eigenvalues[:num_qubits]
            z2 = measurement_eigenvalues[num_qubits:]

            # Implementing the sum and product from the paper
            # each factor in the product of the Ei sum will be either +/-1
            # only in case of pauli Z
            product_term = 1
            for j in range(num_qubits):
                product_term *= (1 + z1[j] - z2[j] + z1[j] * z2[j]) // 2

            D += product_term # correct for M=2 case, but wrong otherwise

            for i in range(num_qubits):
                Ei[i] += (
                    (z1[i] + z2[i])
                    // 2
                    * product_term
                    // ((1 + z1[i] - z2[i] + z1[i] * z2[i]) // 2)
                )  # undo the j=i term in the product

        expectation_values = Ei / D

    if not np.allclose(expectation_values.real, expectation_values, atol=1e-6):
        warnings.warn(
            "Warning: The expectation value contains a significant \
            imaginary part. This should never happen."
        )
        return expectation_values
    else:
        return expectation_values.real
