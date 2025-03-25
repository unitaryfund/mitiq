from typing import Callable

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
) -> list[MeasurementResult]:
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
    M: int = 2,
    K: int = 1000,
    observable: Observable | None = None,
) -> list[float]:
    """Given a circuit that acts on N qubits, this function returns the
    expectation values of a given observable for each qubit i.
    The expectation values are corrected using the virtual distillation
    algorithm.

    Args:
        circuit: The input circuit of N qubits to execute with VD.
        executor: A Mitiq executor that executes a circuit and returns the
            unmitigated ``QuantumResult`` (e.g. an expectation value).
            The executor must either return a ``MeasurementResult`` or a list
            thereof ``list[MeasurementResult]``.
        M: The number of copies of rho. Only M=2 is implemented at this moment.
        K: The number of iterations of the algorithm. Only used if the executor
            returns a single measurement.
        observable: The one qubit observable for which the expectation values
            are computed. The default observable is the Pauli Z matrix. At the
            moment using different observables is not supported.

    Returns:
        A list of VD-estimated expectation values.
    """

    # input rho is an N qubit circuit
    N = len(circuit.all_qubits())
    new_circuit = _copy_circuit_parallel(circuit, M)

    Ei = np.array(list(0 for _ in range(N)))
    D = 0

    # Forcing odd K, this is a workaround so that D (see end of the function)
    # cannot be 0 accidentally
    if K % 2 == 0:
        K -= 1

    executor_obj = Executor(executor)  # type: ignore[arg-type]

    # Density matrix return type
    if executor_obj._executor_return_type in DensityMatrixLike:
        circuit_dm = executor_obj.run(new_circuit)
        circuit_swaps = _apply_cyclic_system_permutation(circuit_dm, N)
        resulting_dm = _apply_symmetric_observable(
            circuit_swaps, N, observable
        )
        exp_values = np.trace(resulting_dm, axis1=1, axis2=2) / np.trace(
            resulting_dm, axis1=1, axis2=2
        )

    # Measurement result return type
    elif executor_obj._executor_return_type in MeasurementResultLike:
        new_circuit = _apply_diagonalizing_gate(circuit, M)

        # apply measurements
        for i in range(M * N):
            new_circuit.append(cirq.measure(cirq.LineQubit(i), key=f"{i}"))

        res = executor_obj.run(
            new_circuit, force_run_all=True, reps=K
        )  # TODO make this reps a **kwargs to allow any executor

        self_packed = True
        if isinstance(res, str):
            res = [res]
            self_packed = False

        elif isinstance(res[0], int):
            res = [res]
            self_packed = False

        if len(res) == 1:  # if the executor only returns a single measurement
            for _ in range(K - 1):  # then we measure K times in total
                if not self_packed:
                    res.append(
                        executor_obj.run(new_circuit, force_run_all=True)
                    )
                else:
                    res.append(
                        executor_obj.run(new_circuit, force_run_all=True)[0]
                    )

        # post processing measurements
        for bitStr in res:
            # map 0/1 measurements to 1/-1 measurements (eigenvalues of Z)
            Z_base_measurement = np.array(
                1 if i == 0 else -1 for i in list(bitStr)
            )

            # Separate the two systems
            z1 = Z_base_measurement[:N]
            z2 = Z_base_measurement[N:]

            # Implementing the sum and product from the paper
            # each factor in the product of the Ei sum will be either +/-1
            # only in case of pauli Z
            product_term = 1
            for j in range(N):
                product_term *= (1 + z1[j] - z2[j] + z1[j] * z2[j]) // 2

            D += product_term

            for i in range(N):
                Ei[i] += (
                    (z1[i] + z2[i])
                    // 2
                    * product_term
                    // ((1 + z1[i] - z2[i] + z1[i] * z2[i]) // 2)
                )  # undo the j=i term in the product

        exp_values = Ei / D

    else:
        raise ValueError(
            "Executor must have a return type of DensityMatrixLike or \
            MeasurementResultLike"
        )

    if not np.allclose(exp_values.real, exp_values, atol=1e-6):
        print(
            "Warning: The expectation value contains a significant \
            imaginary part. This should never happen."
        )
        return exp_values
    else:
        return exp_values.real
